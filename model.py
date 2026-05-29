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
                u_all, i_all, v_norm = self._build_graphaug_view2_edges(
                    self.E_u.detach(),
                    self.E_i.detach(),
                )

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

            # -------------------------
            # DirectAU Loss (Replaces InfoNCE)
            # -------------------------
            # 1. Normalize embeddings to the unit hypersphere (L2 normalization)
            G_u_norm = F.normalize(self.G_u, p=2, dim=1)
            E_u_norm = F.normalize(self.E_u, p=2, dim=1)
            G_i_norm = F.normalize(self.G_i, p=2, dim=1)
            E_i_norm = F.normalize(self.E_i, p=2, dim=1)

            # 2. Alignment Loss (Distance between positive views)
            align_u = (G_u_norm[uids] - E_u_norm[uids]).norm(p=2, dim=1).pow(2).mean()
            align_i = (G_i_norm[iids] - E_i_norm[iids]).norm(p=2, dim=1).pow(2).mean()
            loss_align = align_u + align_i

            # 3. Uniformity Loss (Pushing batch nodes apart to avoid collapse)
            t = 2.0  # Hyperparameter controlling uniformity strength
            uniform_u = torch.pdist(E_u_norm[uids], p=2).pow(2).mul(-t).exp().mean().log()
            uniform_i = torch.pdist(E_i_norm[iids], p=2).pow(2).mul(-t).exp().mean().log()
            loss_uniform = uniform_u + uniform_i

            # 4. Combine Alignment and Uniformity for final SSL loss
            gamma = 1.0  # Weight for Uniformity
            loss_s = loss_align + gamma * loss_uniform

            # -------------------------
            # BPR Loss (Recommendation Task)
            # -------------------------
            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

            # -------------------------
            # Reg Loss (Weight Decay)
            # -------------------------
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

            # Total loss
            loss = loss_r + self.lambda_1 * loss_s + loss_reg
            return loss, loss_r, self.lambda_1 * loss_s

    def _build_graphaug_view2_edges(self, E_u_detached: torch.Tensor, E_i_detached: torch.Tensor):
        assert self.obs_u is not None and self.obs_i is not None, 'obs_u/obs_i must be provided for graphaug view2'
        assert self.twohop_u is not None and self.twohop_i is not None, 'twohop_u/twohop_i must be provided for graphaug view2'

        # Predict probabilities for 2-hop candidate edges
        hu = E_u_detached[self.twohop_u]
        hv = E_i_detached[self.twohop_i]
        logits = self.aug_mlp(torch.cat([hu, hv], dim=1)).squeeze(-1)
        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)

        # Gumbel-sigmoid (Concrete) sampling
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