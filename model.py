import torch
import torch.nn as nn
from utils import sparse_dropout, spmm
import torch.nn.functional as F

class LightGCL(nn.Module):
    def __init__(
        self,
        n_u,
        n_i,
        d,
        train_csr,
        adj_norm,
        l,
        temp,
        lambda_1,
        lambda_2,
        dropout,
        batch_user,
        device,
        # view2 options
        view2_mode='graphaug',
        # svd-view parameters (kept for backward compatibility)
        u_mul_s=None,
        v_mul_s=None,
        ut=None,
        vt=None,
        # graphaug-view parameters
        obs_u=None,
        obs_i=None,
        twohop_u=None,
        twohop_i=None,
        tau1=0.5,
        xi=0.5,
        aug_hidden=None,
    ):
        super(LightGCL,self).__init__()
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d)))

        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l+1)
        self.Z_i_list = [None] * (l+1)
        self.G_u_list = [None] * (l+1)
        self.G_i_list = [None] * (l+1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user

        self.E_u = None
        self.E_i = None

        self.view2_mode = view2_mode

        # SVD-view params
        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        # GraphAug-view params (Option B: observed edges + sampled 2-hop candidates)
        # obs_u/obs_i and twohop_u/twohop_i are 1D Long tensors on GPU.
        self.obs_u = obs_u
        self.obs_i = obs_i
        self.twohop_u = twohop_u
        self.twohop_i = twohop_i
        self.tau1 = tau1
        self.xi = xi

        h = aug_hidden if aug_hidden is not None else d
        # A small edge-probability predictor: p(u,v)=sigmoid(MLP([h_u||h_v]))
        self.aug_mlp = nn.Sequential(
            nn.Linear(2 * d, h),
            nn.LeakyReLU(0.2),
            nn.Linear(h, 1),
        )

        self.device = device

    def forward(self, uids, iids, pos, neg, test=False):
        if test==True:  # testing phase
            preds = self.E_u[uids] @ self.E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).cuda(torch.device(self.device))
            preds = preds * (1-mask) - 1e8 * mask
            predictions = preds.argsort(descending=True)
            return predictions
        else:  # training phase
            # -------------------------
            # View 1: original graph (LightGCL)
            # -------------------------
            for layer in range(1, self.l + 1):
                # GNN propagation (edge dropout only for view1)
                A_drop = sparse_dropout(self.adj_norm, self.dropout)
                self.Z_u_list[layer] = torch.spmm(A_drop, self.E_i_list[layer - 1])
                self.Z_i_list[layer] = torch.spmm(A_drop.transpose(0, 1), self.E_u_list[layer - 1])

                # aggregate
                self.E_u_list[layer] = self.Z_u_list[layer]
                self.E_i_list[layer] = self.Z_i_list[layer]

            # aggregate across layers (view1 embeddings)
            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)

            # -------------------------
            # View 2: either SVD-view (original) or GraphAug-view (new)
            # -------------------------
            if self.view2_mode == 'svd':
                # original LightGCL SVD view
                for layer in range(1, self.l + 1):
                    vt_ei = self.vt @ self.E_i_list[layer - 1]
                    self.G_u_list[layer] = self.u_mul_s @ vt_ei
                    ut_eu = self.ut @ self.E_u_list[layer - 1]
                    self.G_i_list[layer] = self.v_mul_s @ ut_eu
                self.G_u = sum(self.G_u_list)
                self.G_i = sum(self.G_i_list)
            else:
                # GraphAug view2: observed edges + sampled 2-hop candidates.
                # Stop-gradient: detach embeddings used by the augmentor.
                # IMPORTANT: torch.spmm does NOT support backprop w.r.t. sparse values.
                # Since A' edge weights are produced by the augmentor and require gradients,
                # we implement message passing using edge lists (index_add), which supports
                # gradients w.r.t. edge weights.
                u_all, i_all, v_norm = self._build_graphaug_view2_edges(
                    self.E_u.detach(),
                    self.E_i.detach(),
                )

                # IMPORTANT (LightGCL-style View2):
                # In the original LightGCL, the auxiliary (SVD) view at layer `l` is computed
                # from the *main-view embeddings at layer l-1* (E_{l-1}), not by recursively
                # propagating within the auxiliary view itself.
                #
                # To keep the workflow consistent, we compute:
                #   G_l = A' @ E_{l-1}
                # rather than:
                #   G_l = A' @ G_{l-1}
                #
                # This typically improves stability and reduces view mismatch that can
                # degrade Recall/NDCG.
                self.G_u_list = [None] * (self.l + 1)
                self.G_i_list = [None] * (self.l + 1)
                self.G_u_list[0] = self.E_u_0
                self.G_i_list[0] = self.E_i_0
                for layer in range(1, self.l + 1):
                    gu, gi = self._propagate_bipartite_edges(
                        self.E_u_list[layer - 1],
                        self.E_i_list[layer - 1],
                        u_all,
                        i_all,
                        v_norm,
                    )
                    self.G_u_list[layer] = gu
                    self.G_i_list[layer] = gi
                self.G_u = sum(self.G_u_list)
                self.G_i = sum(self.G_i_list)

            # cl loss
            G_u_norm = self.G_u
            E_u_norm = self.E_u
            G_i_norm = self.G_i
            E_i_norm = self.E_i
            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
            loss_s = -pos_score + neg_score

            # bpr loss
            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

            # total loss
            loss = loss_r + self.lambda_1 * loss_s + loss_reg
            #print('loss',loss.item(),'loss_r',loss_r.item(),'loss_s',loss_s.item())
            return loss, loss_r, self.lambda_1 * loss_s

    def _build_graphaug_view2_edges(self, E_u_detached: torch.Tensor, E_i_detached: torch.Tensor):
        """Build *edge lists* for view2 using:
        A' = observed_edges (weight=1) U sampled_twohop_edges (weight=soft_sample, thresholded).

        Returns:
            u_all: LongTensor [E] user indices
            i_all: LongTensor [E] item indices
            v_norm: FloatTensor [E] normalized edge weights (requires grad for 2-hop part)

        Notes:
            We intentionally avoid constructing a sparse tensor A' because torch.spmm
            does not support gradients w.r.t. sparse values. Using edge lists + index_add
            supports gradients w.r.t. v_norm, enabling the augmentor to learn.

        - Option B: observed edges + sampled 2-hop candidates.
        - Density control: threshold xi.
        - Stop-gradient: caller passes detached embeddings.
        """
        assert self.obs_u is not None and self.obs_i is not None, 'obs_u/obs_i must be provided for graphaug view2'
        assert self.twohop_u is not None and self.twohop_i is not None, 'twohop_u/twohop_i must be provided for graphaug view2'

        # Predict probabilities for 2-hop candidate edges
        hu = E_u_detached[self.twohop_u]
        hv = E_i_detached[self.twohop_i]
        logits = self.aug_mlp(torch.cat([hu, hv], dim=1)).squeeze(-1)
        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)

        # Gumbel-sigmoid (Concrete) sampling
        # g = -log(-log(u))
        u = torch.rand_like(p)
        g = -torch.log(-torch.log(u + 1e-8) + 1e-8)
        logit_p = torch.log(p) - torch.log(1 - p)
        soft = torch.sigmoid((logit_p + g) / self.tau1)

        # Threshold to control density
        keep = (soft > self.xi).to(soft.dtype)
        v_twohop = soft * keep

        # Observed edges are always kept with weight 1
        u_all = torch.cat([self.obs_u, self.twohop_u], dim=0)
        i_all = torch.cat([self.obs_i, self.twohop_i], dim=0)
        v_all = torch.cat([torch.ones_like(self.obs_u, dtype=soft.dtype), v_twohop], dim=0)

        # Compute bipartite normalization: v / sqrt(deg_u[u] * deg_i[i])
        n_u = self.E_u_0.shape[0]
        n_i = self.E_i_0.shape[0]
        deg_u = torch.zeros(n_u, device=v_all.device, dtype=v_all.dtype)
        deg_i = torch.zeros(n_i, device=v_all.device, dtype=v_all.dtype)
        deg_u.index_add_(0, u_all, v_all)
        deg_i.index_add_(0, i_all, v_all)
        norm = torch.sqrt((deg_u[u_all] * deg_i[i_all]).clamp_min(1e-12))
        v_norm = v_all / norm

        return u_all, i_all, v_norm

    @staticmethod
    def _propagate_bipartite_edges(
        E_u_prev: torch.Tensor,
        E_i_prev: torch.Tensor,
        u_idx: torch.Tensor,
        i_idx: torch.Tensor,
        w: torch.Tensor,
    ):
        """One LightGCN-style propagation step on a bipartite graph defined by edge lists.

        E_u_next[u] = sum_{(u,i) in E} w(u,i) * E_i_prev[i]
        E_i_next[i] = sum_{(u,i) in E} w(u,i) * E_u_prev[u]

        This is equivalent to:
            E_u_next = A' @ E_i_prev
            E_i_next = A'^T @ E_u_prev
        but implemented with index_add so gradients can flow through w.
        """
        n_u, d = E_u_prev.shape
        n_i = E_i_prev.shape[0]

        # user aggregation
        msg_u = E_i_prev[i_idx] * w.unsqueeze(1)
        E_u_next = torch.zeros((n_u, d), device=E_u_prev.device, dtype=E_u_prev.dtype)
        E_u_next.index_add_(0, u_idx, msg_u)

        # item aggregation
        msg_i = E_u_prev[u_idx] * w.unsqueeze(1)
        E_i_next = torch.zeros((n_i, d), device=E_i_prev.device, dtype=E_i_prev.dtype)
        E_i_next.index_add_(0, i_idx, msg_i)

        return E_u_next, E_i_next
