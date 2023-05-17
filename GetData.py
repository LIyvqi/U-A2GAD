import dgl
from dgl.data import FraudYelpDataset, FraudAmazonDataset,CoraGraphDataset,PubmedGraphDataset
from dgl.data.utils import save_graphs,load_graphs
import numpy as np


class Dataset:
    def __init__(self, name='yelp', homo=True, anomaly_alpha=None, anomaly_std=None):
        self.name = name
        graph = None
        if name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                # graph = dgl.add_self_loop(graph)
        elif name == 'elliptic':
            graph = load_graphs("../dataSet/elliptic_dgl_g.bin")
            graph = graph[0][0]
        elif name == 'cora':
            dataset = CoraGraphDataset()
            graph = dgl.to_homogeneous(dataset[0], ndata=['feat', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)
            labels = graph.ndata['label']
            normal_idx = np.where(labels != 6)[0]
            abnormal_idx = np.where(labels == 6)[0]
            labels[normal_idx] = 0
            labels[abnormal_idx] = 1
            graph.ndata['label'] = labels
            graph.ndata['feature'] = graph.ndata['feat']
        elif name == 'Pubmed':
            dataset = PubmedGraphDataset()
            graph = dgl.to_homogeneous(dataset[0], ndata=['feat', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)
            labels = graph.ndata['label']
            normal_idx = np.where(labels != 0)[0]
            abnormal_idx = np.where(labels == 0)[0]
            labels[normal_idx] = 0
            labels[abnormal_idx] = 1
            graph.ndata['label'] = labels
            graph.ndata['feature'] = graph.ndata['feat']


        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()

        self.graph = graph