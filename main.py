import os
import random
from GetData import *
from Train import *
from models import *


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='yelp', help='dataset name')
parser.add_argument('--model', type=str, default='AdaDGNN_neg', help='model')
parser.add_argument('--topk', type=int, default=4, help='topk nums')
parser.add_argument('--neg_topk', type=int, default=150, help='neg topk nums')
parser.add_argument('--hid_dim', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--drop_ratio', type=float, default=0.1)
parser.add_argument('--train_ratio', type=float, default=0.4)
parser.add_argument('--epoch', type=int, default=2000)
parser.add_argument('--wd', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--run', type=int, default=1)
parser.add_argument('--order', type=int, default=2)
parser.add_argument('--homo', type=int, default=1)
args = parser.parse_args()

def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_torch(seed=args.seed)

def main(args):
    dataset_name = args.dataset
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim
    graph = Dataset(dataset_name, homo).graph

    from dgl.nn.pytorch.factory import KNNGraph

    in_feats = graph.ndata['feature'].shape[1]
    num_classes = 2

    print(args.dataset)
    print(graph)

    kg = KNNGraph(args.topk)
    g = kg(graph.ndata['feature'])
    g = dgl.to_bidirected(g)
    g.ndata['feature'] = graph.ndata['feature']

    neg_g = neg_knn_graph(graph.ndata['feature'], args.neg_topk, exclude_self=True)
    neg_g = dgl.to_bidirected(neg_g)
    neg_g.ndata['feature'] = graph.ndata['feature']

    model = AdaDGNN_neg(in_feats, h_feats, num_classes, graph, g, neg_g, args, d=order)
    mf1, auc, auc_pr = train(model, graph, args)

    print("Final result:",mf1,auc,auc_pr)


main(args)
