import os
import pickle
import time

import networkx as nx
import numpy as np
import scipy.sparse as spsprs
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import sklearn.metrics
import scipy.io as scio

from sklearn.cluster import KMeans

from data.io import load_dataset
from data.sparsegraph import SparseGraph

from network import *
from utils import *
from preprocessing import gen_splits, normalize_attributes


class NeibSampler2:
    def __init__(self, graph, adj, nb_size, include_self=False):
        n = graph.number_of_nodes()
        self.adj = adj
        assert 0 <= min(graph.nodes()) and max(graph.nodes()) < n
        
        if include_self:
            nb_all = torch.zeros(n, nb_size + 1, dtype=torch.int64)
            nb_all[:, 0] = torch.arange(0, n)
            nb = nb_all[:, 1:]
        else:
            nb_all = torch.zeros(n, nb_size, dtype=torch.int64)
            nb = nb_all
       
        popkids = []
        for v in range(n):
            nb_v = np.sort(np.nonzero(adj[v])).tolist()[0]
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))
                nb[v] = torch.LongTensor(nb_v)
            else:
                popkids.append(v)
        self.include_self = include_self
        self.g, self.nb_all, self.pk = graph, nb_all, popkids
        self.n, self.nb_size = n, nb_size
        self.sortV_list = []
        self.p_list = []

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def update(self, adj):
        n, nb_size = self.n, self.nb_size
        popkids = []
        
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all
        
        
        for v in range(n):
            nb_v = np.sort(np.nonzero(adj[v])).tolist()[0]
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))
                nb[v] = torch.LongTensor(nb_v)
            else:
                popkids.append(v)      
        self.pk = popkids
        
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all
        nb_size = nb.size(1)
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)
        sortV_list = []
        p_list = []
        for i, v in enumerate(self.pk):
            sortV = np.sort(np.nonzero(adj[v])).tolist()[0]
            sortV_list.append(sortV)
            neighborSet = sortV.copy()
            
            p = np.array([adj[v][j] for j in sortV])
            p = p/sum(p)
            p_list.append(p)
            
            pk_nb[i] = np.random.choice(neighborSet, nb_size, p = p)
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        self.sortV_list = sortV_list
        self.p_list = p_list
    
    def sample(self):
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all
        nb_size = nb.size(1)
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)
        if self.sortV_list!= []:
            for i, v in enumerate(self.pk):
                sortV = self.sortV_list[i]
                neighborSet = sortV.copy()
                p = self.p_list[i]
                pk_nb[i] = np.random.choice(neighborSet, nb_size, p = p)
                
        elif self.sortV_list == []:
            for i, v in enumerate(self.pk):
                pk_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), nb_size)    
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        #print(np.nonzero(adj[v]))
        #print(len(np.nonzero(adj[v])[0]),'the last v',v)
        return self.nb_all


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


class EvalHelper:
    # noinspection PyUnresolvedReferences
    def __init__(self, dataset, hyperpm):
        self.hyperpm = hyperpm
        use_cuda = torch.cuda.is_available() and not hyperpm.cpu
        dev = torch.device('cuda' if use_cuda else 'cpu')
        graph, feat, targ = dataset.get_graph_feat_targ()
        targ = torch.from_numpy(targ).to(dev).long()
        feat = thsprs_from_spsprs(feat).to(dev)
        trn_idx, val_idx, tst_idx = dataset.get_split()
        trn_idx = torch.from_numpy(trn_idx).to(dev)
        val_idx = torch.from_numpy(val_idx).to(dev)
        tst_idx = torch.from_numpy(tst_idx).to(dev)
        nfeat, nclass = feat.size(1), int(targ.max() + 1)
        model = CapsuleNet(nfeat, nclass, hyperpm).to(dev)
        var_list = [var for name, var in model.named_parameters() if 'dis' not in name]
        optmz = optim.Adam(var_list, lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.graph, self.feat, self.targ = graph, feat, targ
        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        self.model, self.optmz, self.dev = model, optmz, dev
        self.ncaps = hyperpm.ncaps
        self.adj = dataset.get_graph_adj()
        self.updated_adj = self.adj.copy()
        self.neib_sampler = NeibSampler2(graph, self.adj, hyperpm.nbsz).to(dev)
        self.index = []
        self.F = torch.tensor(self.feat.cpu().to_dense().numpy())
        self.task = hyperpm.task
        self.n_cluster = hyperpm.n_cluster


    def run_epoch(self, flag = False, end='\n'):
        start = time.time()
        self.model.train()
        self.optmz.zero_grad()
        prob, GenLossList, embedding = self.model(self.feat, self.neib_sampler.sample())
        loss = fn.nll_loss(prob[self.trn_idx], self.targ[self.trn_idx])
        adv_loss, d_loss = GenLossList[0], GenLossList[1]
        total_loss = adv_loss + loss
        total_loss.backward()
        self.optmz.step()

        self.model.optmzD.zero_grad()
        d_loss.backward()
        self.model.optmzD.step()
        
        print('trn-loss: %.4f, adv-loss: %.4f' % (loss.item(),adv_loss.item()), end=end)
        if flag != False:
            self.index = Soft_Degree_Sampling_prob(self.graph, embedding)
            self.updated_adj = graph_refine(embedding, self.adj, rate = self.hyperpm.ratio, ncaps = self.ncaps, node_index=self.index)
            self.neib_sampler.update(self.updated_adj)
        end = time.time()
        tc = end - start
        return loss.item(), tc

    def print_trn_acc(self):
        print('trn-', end='')
        trn_acc, f1, _, _ = self._print_acc(self.trn_idx, end=' val-')
        val_acc, f1, _, _ = self._print_acc(self.val_idx)
        return trn_acc, val_acc

    def print_tst_acc(self):
        print('tst-', end='')
        tst_acc, tst_f1, embedding, targ = self._print_acc(self.tst_idx, mode = 'test')
        embedding = embedding.detach().cpu().numpy()
        targ = targ.detach().cpu().numpy()

        #embedding = temp[0]
        #targ = temp[1]
        tst_f1 = self._print_f1(self.tst_idx)
        return tst_acc, tst_f1

    def _print_acc(self, eval_idx, mode=None, end='\n'):
        self.model.eval()
        output = self.model(self.feat, self.neib_sampler.nb_all)
        prob = output[0][eval_idx]
        targ = self.targ[eval_idx]
        embedding = output[2]
        pred = prob.max(1)[1].type_as(targ)
        acc = pred.eq(targ).double().sum() / len(targ)
        acc = acc.item()
        
        targ = targ.cpu().numpy()
        pred = pred.cpu().numpy()
        f1 = sklearn.metrics.f1_score(targ, pred, average='macro')
        embedding = output[2]
        embed = embedding.cpu().detach().numpy()
        if mode == 'test':
            pass
            #np.save('./save/{}-ADGCN_total'.format(self.hyperpm.datname), embed)
        if self.task == 'clustering':
            kmeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(embed)
            predict_labels = kmeans.predict(embed)
            eval_idx = eval_idx.cpu().detach().numpy()
            cm = clustering_metrics(predict_labels[eval_idx], self.targ[eval_idx].cpu().detach().numpy())
            acc_cluster, nmi, adjscore = cm.evaluationClusterModelFromLabel(mode)
            acc = nmi
        
        print('acc: %.4f' % acc, end=end)
        return acc, f1, embedding, self.targ
    
    def _print_f1(self, eval_idx, end='\n'):
        self.model.eval()
        prob = self.model(self.feat, self.neib_sampler.nb_all)[0][eval_idx]
        targ = self.targ[eval_idx]
        pred = prob.max(1)[1].type_as(targ).cpu().numpy()
        targ = targ.cpu().numpy()
        f1 = sklearn.metrics.f1_score(targ, pred, average='macro')
        print('f1: %.4f' % f1, end=end)
        #print(sklearn.metrics.classification_report(targ, pred))
        return f1

    def visualize(self, sav_prefix):
        g = self.graph
        n = g.number_of_nodes()
        self.model.eval()
        prob = self.model(self.feat, self.neib_sampler.nb_all)[0].cpu()
        targ = self.targ.cpu()
        pred = prob.max(1)[1].type_as(targ).eq(targ).numpy()
        acc = [('.' if c else '?') for c in pred.astype(dtype=np.bool)]
        sets = np.zeros(n, dtype=np.float32)
        sets[self.trn_idx.cpu()] = 0
        sets[self.val_idx.cpu()] = 1
        sets[self.tst_idx.cpu()] = 2
        pos_gml = sav_prefix + '.gml'
        visualize_as_gdf(g, sav_prefix, list(range(n)), targ, pos_gml)
        visualize_as_gdf(g, sav_prefix + '_set', pred, sets, pos_gml)
        visualize_as_gdf(g, sav_prefix + '_trg', pred, targ, pos_gml)