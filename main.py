import numpy as np
import torch
import pickle
import os
import random
from model import LightGCL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
import pandas as pd
from parser import args
from tqdm import tqdm
import time
import torch.utils.data as data
from utils import TrnData
from collections import defaultdict

device = 'cuda:' + args.cuda


def seed_everything(seed: int, deterministic: bool = False):
    """Seed python / numpy / torch for reproducibility.

    Notes:
      - Determinism on GPU is best-effort; some sparse ops may still be non-deterministic
        depending on your PyTorch/CUDA versions.
      - Setting deterministic=True may reduce performance.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Best-effort deterministic algorithms (may raise if an op has no deterministic kernel)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        # For CUDA matmul determinism (optional; harmless if unsupported)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


# Apply seed as early as possible
seed_everything(args.seed, deterministic=args.deterministic)

# hyperparameters
d = args.d
l = args.gnn_layer
temp = args.temp
batch_user = args.batch
epoch_no = args.epoch
max_samp = 40
lambda_1 = args.lambda1
lambda_2 = args.lambda2
dropout = args.dropout
lr = args.lr
decay = args.decay
svd_q = args.q
view2_mode = args.view2

# load data
path = 'data/' + args.data + '/'
f = open(path+'trnMat.pkl','rb')
train = pickle.load(f)
train_csr = (train!=0).astype(np.float32)
f = open(path+'tstMat.pkl','rb')
test = pickle.load(f)
print('Data loaded.')

print('user_num:',train.shape[0],'item_num:',train.shape[1],'lambda_1:',lambda_1,'lambda_2:',lambda_2,'temp:',temp,'q:',svd_q)

epoch_user = min(train.shape[0], 30000)

# normalizing the adj matrix
rowD = np.array(train.sum(1)).squeeze()
colD = np.array(train.sum(0)).squeeze()
for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)

# construct data loader
train = train.tocoo()
train_data = TrnData(train)
# Make DataLoader shuffling reproducible
_dl_gen = torch.Generator()
_dl_gen.manual_seed(args.seed)
train_loader = data.DataLoader(
    train_data,
    batch_size=args.inter_batch,
    shuffle=True,
    num_workers=0,
    generator=_dl_gen,
)

adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce().cuda(torch.device(device))
print('Adj matrix normalized.')

def build_twohop_candidates(train_csr_mat, train_csc_mat, twohop_per_user, seed_items, users_per_item, item_sample, rng):
    """Option B: observed edges + sampled 2-hop candidates.

    For each user u:
      1) sample `seed_items` items from N(u)
      2) for each seed item, sample `users_per_item` users from its neighbors
      3) sample up to `item_sample` items from each neighbor user, union them
      4) remove observed items and keep up to `twohop_per_user`

    Returns:
      twohop_u, twohop_i: numpy arrays of edge endpoints for candidate edges.
    """
    n_u = train_csr_mat.shape[0]
    twohop_u = []
    twohop_i = []

    for u in range(n_u):
        items_u = train_csr_mat.indices[train_csr_mat.indptr[u]:train_csr_mat.indptr[u+1]]
        if len(items_u) == 0:
            continue

        # sample seed items from user's observed items
        if len(items_u) <= seed_items:
            seed = items_u
        else:
            seed = rng.choice(items_u, size=seed_items, replace=False)

        cand_items = set()
        obs_set = set(items_u.tolist())

        for it in seed:
            users_it = train_csc_mat.indices[train_csc_mat.indptr[it]:train_csc_mat.indptr[it+1]]
            if len(users_it) == 0:
                continue
            if len(users_it) <= users_per_item:
                neigh_users = users_it
            else:
                neigh_users = rng.choice(users_it, size=users_per_item, replace=False)

            for nu in neigh_users:
                items_nu = train_csr_mat.indices[train_csr_mat.indptr[nu]:train_csr_mat.indptr[nu+1]]
                if len(items_nu) == 0:
                    continue
                if len(items_nu) > item_sample:
                    items_nu = rng.choice(items_nu, size=item_sample, replace=False)
                cand_items.update(items_nu.tolist())

        # remove observed items
        cand_items.difference_update(obs_set)
        if len(cand_items) == 0:
            continue

        # keep at most twohop_per_user
        if len(cand_items) > twohop_per_user:
            cand_items = rng.choice(np.array(list(cand_items), dtype=np.int64), size=twohop_per_user, replace=False)
        else:
            cand_items = np.array(list(cand_items), dtype=np.int64)

        twohop_u.append(np.full(len(cand_items), u, dtype=np.int64))
        twohop_i.append(cand_items)

    if len(twohop_u) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    return np.concatenate(twohop_u), np.concatenate(twohop_i)


# Build view2 inputs
if view2_mode == 'svd':
    # perform svd reconstruction (original LightGCL)
    adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cuda(torch.device(device))
    print('Performing SVD...')
    svd_u, s, svd_v = torch.svd_lowrank(adj, q=svd_q)
    u_mul_s = svd_u @ (torch.diag(s))
    v_mul_s = svd_v @ (torch.diag(s))
    del s
    print('SVD done.')
    obs_u_t = obs_i_t = twohop_u_t = twohop_i_t = None
else:
    # GraphAug view2: observed edges + sampled 2-hop candidates
    print('Building 2-hop candidates for GraphAug view2...')
    # Use unnormalized binary adjacency for neighbor queries
    train_bin = train_csr.copy()
    train_bin.data = np.ones_like(train_bin.data)
    csr_bin = train_bin.tocsr()
    csc_bin = train_bin.tocsc()
    rng = np.random.default_rng(seed=args.seed)

    twohop_u_np, twohop_i_np = build_twohop_candidates(
        csr_bin,
        csc_bin,
        twohop_per_user=args.twohop_per_user,
        seed_items=args.twohop_seed_items,
        users_per_item=args.twohop_users_per_item,
        item_sample=args.twohop_item_sample,
        rng=rng,
    )

    # observed edges from COO
    obs_u_np = train.row.astype(np.int64)
    obs_i_np = train.col.astype(np.int64)
    print(f'Observed edges: {len(obs_u_np):,} | 2-hop candidate edges: {len(twohop_u_np):,}')

    obs_u_t = torch.from_numpy(obs_u_np).long().cuda(torch.device(device))
    obs_i_t = torch.from_numpy(obs_i_np).long().cuda(torch.device(device))
    twohop_u_t = torch.from_numpy(twohop_u_np).long().cuda(torch.device(device))
    twohop_i_t = torch.from_numpy(twohop_i_np).long().cuda(torch.device(device))
    u_mul_s = v_mul_s = svd_u = svd_v = None

# process test set
test_labels = [[] for i in range(test.shape[0])]
for i in range(len(test.data)):
    row = test.row[i]
    col = test.col[i]
    test_labels[row].append(col)
print('Test data processed.')

loss_list = []
loss_r_list = []
loss_s_list = []
recall_20_x = []
recall_20_y = []
ndcg_20_y = []
recall_40_y = []
ndcg_40_y = []

model = LightGCL(
    adj_norm.shape[0],
    adj_norm.shape[1],
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
    view2_mode=view2_mode,
    # svd params
    u_mul_s=u_mul_s,
    v_mul_s=v_mul_s,
    ut=(svd_u.T if view2_mode == 'svd' else None),
    vt=(svd_v.T if view2_mode == 'svd' else None),
    # graphaug params
    obs_u=obs_u_t,
    obs_i=obs_i_t,
    twohop_u=twohop_u_t,
    twohop_i=twohop_i_t,
    tau1=args.tau1,
    xi=args.xi,
    aug_hidden=args.aug_hidden,
)
#model.load_state_dict(torch.load('saved_model.pt'))
model.cuda(torch.device(device))
optimizer = torch.optim.Adam(model.parameters(),weight_decay=0,lr=lr)
#optimizer.load_state_dict(torch.load('saved_optim.pt'))

current_lr = lr

for epoch in range(epoch_no):
    if (epoch+1)%50 == 0:
        torch.save(model.state_dict(),'saved_model/saved_model_epoch_'+str(epoch)+'.pt')
        torch.save(optimizer.state_dict(),'saved_model/saved_optim_epoch_'+str(epoch)+'.pt')

    epoch_loss = 0
    epoch_loss_r = 0
    epoch_loss_s = 0
    train_loader.dataset.neg_sampling()
    for i, batch in enumerate(tqdm(train_loader)):
        uids, pos, neg = batch
        uids = uids.long().cuda(torch.device(device))
        pos = pos.long().cuda(torch.device(device))
        neg = neg.long().cuda(torch.device(device))
        iids = torch.concat([pos, neg], dim=0)

        # feed
        optimizer.zero_grad()
        loss, loss_r, loss_s= model(uids, iids, pos, neg)
        loss.backward()
        optimizer.step()
        #print('batch',batch)
        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        epoch_loss_s += loss_s.cpu().item()

        torch.cuda.empty_cache()
        #print(i, len(train_loader), end='\r')

    batch_no = len(train_loader)
    epoch_loss = epoch_loss/batch_no
    epoch_loss_r = epoch_loss_r/batch_no
    epoch_loss_s = epoch_loss_s/batch_no
    loss_list.append(epoch_loss)
    loss_r_list.append(epoch_loss_r)
    loss_s_list.append(epoch_loss_s)
    print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_r:',epoch_loss_r,'Loss_s:',epoch_loss_s)

    if epoch % 3 == 0:  # test every 10 epochs
        test_uids = np.array([i for i in range(adj_norm.shape[0])])
        batch_no = int(np.ceil(len(test_uids)/batch_user))

        all_recall_20 = 0
        all_ndcg_20 = 0
        all_recall_40 = 0
        all_ndcg_40 = 0
        for batch in tqdm(range(batch_no)):
            start = batch*batch_user
            end = min((batch+1)*batch_user,len(test_uids))

            test_uids_input = torch.LongTensor(test_uids[start:end]).cuda(torch.device(device))
            predictions = model(test_uids_input,None,None,None,test=True)
            predictions = np.array(predictions.cpu())

            #top@20
            recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
            #top@40
            recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

            all_recall_20+=recall_20
            all_ndcg_20+=ndcg_20
            all_recall_40+=recall_40
            all_ndcg_40+=ndcg_40
            #print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)
        print('-------------------------------------------')
        print('Test of epoch',epoch,':','Recall@20:',all_recall_20/batch_no,'Ndcg@20:',all_ndcg_20/batch_no,'Recall@40:',all_recall_40/batch_no,'Ndcg@40:',all_ndcg_40/batch_no)
        recall_20_x.append(epoch)
        recall_20_y.append(all_recall_20/batch_no)
        ndcg_20_y.append(all_ndcg_20/batch_no)
        recall_40_y.append(all_recall_40/batch_no)
        ndcg_40_y.append(all_ndcg_40/batch_no)

# final test
test_uids = np.array([i for i in range(adj_norm.shape[0])])
batch_no = int(np.ceil(len(test_uids)/batch_user))

all_recall_20 = 0
all_ndcg_20 = 0
all_recall_40 = 0
all_ndcg_40 = 0
for batch in range(batch_no):
    start = batch*batch_user
    end = min((batch+1)*batch_user,len(test_uids))

    test_uids_input = torch.LongTensor(test_uids[start:end]).cuda(torch.device(device))
    predictions = model(test_uids_input,None,None,None,test=True)
    predictions = np.array(predictions.cpu())

    #top@20
    recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
    #top@40
    recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

    all_recall_20+=recall_20
    all_ndcg_20+=ndcg_20
    all_recall_40+=recall_40
    all_ndcg_40+=ndcg_40
    #print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)
print('-------------------------------------------')
print('Final test:','Recall@20:',all_recall_20/batch_no,'Ndcg@20:',all_ndcg_20/batch_no,'Recall@40:',all_recall_40/batch_no,'Ndcg@40:',all_ndcg_40/batch_no)

recall_20_x.append('Final')
recall_20_y.append(all_recall_20/batch_no)
ndcg_20_y.append(all_ndcg_20/batch_no)
recall_40_y.append(all_recall_40/batch_no)
ndcg_40_y.append(all_ndcg_40/batch_no)

metric = pd.DataFrame({
    'epoch':recall_20_x,
    'recall@20':recall_20_y,
    'ndcg@20':ndcg_20_y,
    'recall@40':recall_40_y,
    'ndcg@40':ndcg_40_y
})
current_t = time.gmtime()
metric.to_csv('log/result_'+args.data+'_'+time.strftime('%Y-%m-%d-%H',current_t)+'.csv')

torch.save(model.state_dict(),'saved_model/saved_model_'+args.data+'_'+time.strftime('%Y-%m-%d-%H',current_t)+'.pt')
torch.save(optimizer.state_dict(),'saved_model/saved_optim_'+args.data+'_'+time.strftime('%Y-%m-%d-%H',current_t)+'.pt')