import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    # Reproducibility
    parser.add_argument('--seed', default=2024, type=int, help='random seed (numpy/torch/python)')
    parser.add_argument('--deterministic', action='store_true',
                        help='enable more deterministic behavior (may reduce speed; some ops may still be non-deterministic)')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.99, type=float, help='learning rate')
    parser.add_argument('--batch', default=256, type=int, help='batch size')
    parser.add_argument('--inter_batch', default=4096, type=int, help='batch size')
    parser.add_argument('--note', default=None, type=str, help='note')
    parser.add_argument('--lambda1', default=0.2, type=float, help='weight of cl loss')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--d', default=64, type=int, help='embedding size')
    parser.add_argument('--q', default=5, type=int, help='rank')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
    parser.add_argument('--dropout', default=0.0, type=float, help='rate for edge dropout')
    parser.add_argument('--temp', default=0.2, type=float, help='temperature in cl loss')
    parser.add_argument('--lambda2', default=1e-7, type=float, help='l2 reg weight')
    parser.add_argument('--cuda', default='0', type=str, help='the gpu to use')

    # GraphAug-as-View2 (replace SVD view)
    parser.add_argument('--view2', default='graphaug', type=str, choices=['graphaug','svd'],
                        help='which view2 to use: graphaug (default) or svd (original LightGCL)')
    parser.add_argument('--tau1', default=0.5, type=float, help='temperature for gumbel-sigmoid sampling in view2 graph')
    parser.add_argument('--xi', default=0.5, type=float, help='threshold to keep sampled 2-hop edges in view2 graph')
    parser.add_argument('--twohop_per_user', default=20, type=int, help='max number of 2-hop candidate items per user')
    parser.add_argument('--twohop_seed_items', default=4, type=int, help='number of seed items sampled from user history for 2-hop')
    parser.add_argument('--twohop_users_per_item', default=4, type=int, help='number of neighbor users sampled per seed item')
    parser.add_argument('--twohop_item_sample', default=50, type=int, help='number of items sampled from each neighbor user')
    parser.add_argument('--aug_hidden', default=64, type=int, help='hidden size of augmentor MLP')
    return parser.parse_args()
args = parse_args()
