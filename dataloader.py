import os
import pickle
import random
import numpy as np
import networkx as nx
import scipy.sparse as spsprs
from utils import *
from preprocessing import gen_splits, normalize_attributes
from data.io import load_dataset
from data.sparsegraph import SparseGraph

    
class DataReader:
    def __init__(self, data_name, data_dir, seed, test):
        # Reading the data...
        tmp = []
        prefix = os.path.join(data_dir, 'ind.%s.' % data_name)
        for suffix in ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']:
            with open(prefix + suffix, 'rb') as fin:
                tmp.append(pickle.load(fin, encoding='latin1'))
        x, y, tx, ty, allx, ally, graph = tmp
        with open(prefix + 'test.index') as fin:
            tst_idx = [int(i) for i in fin.read().split()]
        assert np.sum(x != allx[:x.shape[0], :]) == 0
        assert np.sum(y != ally[:y.shape[0], :]) == 0

        # Spliting the data...
        trn_idx = np.array(range(x.shape[0]), dtype=np.int64)
        val_idx = np.array(range(x.shape[0], allx.shape[0]), dtype=np.int64)
        tst_idx = np.array(tst_idx, dtype=np.int64)
        assert len(trn_idx) == x.shape[0]
        assert len(trn_idx) + len(val_idx) == allx.shape[0]
        assert len(tst_idx) > 0
        assert len(set(trn_idx).intersection(val_idx)) == 0
        assert len(set(trn_idx).intersection(tst_idx)) == 0
        assert len(set(val_idx).intersection(tst_idx)) == 0

        # Building the graph...
        graph = nx.from_dict_of_lists(graph)
        assert min(graph.nodes()) == 0
        n = graph.number_of_nodes()
        assert max(graph.nodes()) + 1 == n
        n = max(n, np.max(tst_idx) + 1)
        for u in range(n):
            graph.add_node(u)
        assert graph.number_of_nodes() == n
        assert not graph.is_directed()

        # Building the feature matrix and the label matrix...
        d, c = x.shape[1], y.shape[1]
        feat_ridx, feat_cidx, feat_data = [], [], []
        allx_coo = allx.tocoo()
        for i, j, v in zip(allx_coo.row, allx_coo.col, allx_coo.data):
            feat_ridx.append(i)
            feat_cidx.append(j)
            feat_data.append(v)
        tx_coo = tx.tocoo()
        for i, j, v in zip(tx_coo.row, tx_coo.col, tx_coo.data):
            feat_ridx.append(tst_idx[i])
            feat_cidx.append(j)
            feat_data.append(v)
        if data_name.startswith('nell.0'):
            isolated = np.sort(np.setdiff1d(range(allx.shape[0], n), tst_idx))
            for i, r in enumerate(isolated):
                feat_ridx.append(r)
                feat_cidx.append(d + i)
                feat_data.append(1)
            d += len(isolated)
        feat = spsprs.csr_matrix((feat_data, (feat_ridx, feat_cidx)), (n, d))
        targ = np.zeros((n, c), dtype=np.int64)
        targ[trn_idx, :] = y
        targ[val_idx, :] = ally[val_idx, :]
        targ[tst_idx, :] = ty
        targ = dict((i, j) for i, j in zip(*np.where(targ)))
        targ = np.array([targ.get(i, -1) for i in range(n)], dtype=np.int64)
        print('#instance x #feature ~ #class = %d x %d ~ %d' % (n, d, c))
        feat = normalize_attributes(feat)
        self.nknown = 1500
        idx_split_args = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': self.nknown, 'seed': seed}
        

        # Storing the data...
        #self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        self.trn_idx, self.val_idx, self.tst_idx = gen_splits(targ, idx_split_args, test=test)
        self.graph, self.feat, self.targ = graph, feat, targ

    def get_split(self):
        # *val_idx* contains unlabeled samples for semi-supervised training.
        return self.trn_idx, self.val_idx, self.tst_idx

    def get_graph_feat_targ(self):
        return self.graph, self.feat, self.targ
    
    def get_graph_adj(self):
        self.adj = np.array(nx.adjacency_matrix(self.graph).todense())
        return self.adj
    
    
    
class DataReader_random_split:
    def __init__(self, dataname, datadir=None, seed = 2144199730, test = False):
        self.dataname = datadir + dataname + '.npz'
        with np.load(self.dataname, allow_pickle=True) as loader:
            loader = dict(loader)
            graph = SparseGraph.from_flat_dict(loader)
        #graph = load_dataset(self.dataname)
        graph.standardize(select_lcc=True)
        labels_all = graph.labels
        self.targ = labels_all
        num_classes =len(np.unique(labels_all))
        if dataname in ['cora', 'citeseer', 'pubmed']:
            self.nknown = 1500
        else:
            self.nknown = 5000
        if dataname in ['cora', 'citeseer', 'pubmed']:
            nstopping = 500
        else:
            nstopping = 30 * num_classes 
        idx_split_args = {'ntrain_per_class': 20, 'nstopping': nstopping, 'nknown': self.nknown, 'seed': seed}
        print('class: ',num_classes ,' train_node: ',20 * num_classes,' val_node:', nstopping)
        #self.feat = matrix_to_torch(normalize_attributes(graph.attr_matrix))
        self.feat = normalize_attributes(graph.attr_matrix)
        self.graph_adj = nx.from_scipy_sparse_matrix(graph.adj_matrix)
        self.trn_idx, self.val_idx, self.tst_idx = gen_splits(labels_all, idx_split_args, test=test)
    
    def get_split(self):
        return self.trn_idx, self.val_idx, self.tst_idx
    
    def get_graph_feat_targ(self):
        return self.graph_adj, self.feat, self.targ
    
    def get_graph_adj(self):
        self.adj = np.array(nx.adjacency_matrix(self.graph_adj).todense())
        return self.adj