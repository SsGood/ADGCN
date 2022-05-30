import numpy as np
import scipy.sparse as spsprs
import torch
import networkx as nx
import torch.autograd
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from utils import *
from network import CapsuleNet
    
    
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

class NeibSampler:
    def __init__(self, graph, nb_size, include_self=False):
        n = graph.number_of_nodes()
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
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))
                nb[v] = torch.LongTensor(nb_v)
            else:
                popkids.append(v)
        self.include_self = include_self
        self.g, self.nb_all, self.pk = graph, nb_all, popkids

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def sample(self):
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all
        nb_size = nb.size(1)
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)
        for i, v in enumerate(self.pk):
            pk_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), nb_size)
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        return self.nb_all
    
    
class EvalHelper:
    # noinspection PyUnresolvedReferences
    def __init__(self, dataset, hyperpm):
        use_cuda = torch.cuda.is_available() and not hyperpm.cpu
        dev = torch.device('cuda' if use_cuda else 'cpu')
        graph, feat, targ = dataset.get_graph_feat_targ()
        targ = torch.from_numpy(targ).to(dev)
        feat = thsprs_from_spsprs(feat).to(dev)
        trn_idx, val_idx, tst_idx = dataset.get_split()
        trn_idx = torch.from_numpy(trn_idx).to(dev)
        val_idx = torch.from_numpy(val_idx).to(dev)
        tst_idx = torch.from_numpy(tst_idx).to(dev)
        nfeat, nclass = feat.size(1), int(targ.max() + 1)
        model = CapsuleNet(nfeat, nclass, hyperpm).to(dev)
        var_list = [var for name, var in model.named_parameters() if 'dis' not in name]
        self.optmz = optim.Adam(var_list,lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.optmzD = optim.Adam(model.discriminator.parameters(), lr=0.0002)
        self.graph, self.feat, self.targ, self.adj = graph, feat, targ, dataset.get_graph_adj()
        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        self.model = model
        self.neib_sampler = NeibSampler2(graph, self.adj, hyperpm.nbsz).to(dev)
        self.hyperpm = hyperpm

    def run_epoch(self, flag = False, end='\n'):
        self.model.train()
        self.optmz.zero_grad()
        prob, adv_loss, embedding = self.model(self.feat, self.neib_sampler.sample())
        loss = fn.nll_loss(prob[self.trn_idx], self.targ[self.trn_idx])
        g_loss, d_loss = adv_loss[0], adv_loss[1]
        total_loss = 0.1 * g_loss + loss
        total_loss.backward()
        self.optmz.step()
        
        self.optmzD.zero_grad()
        d_loss.backward()
        self.optmzD.step()
        
        print('trn-loss: %.4f' % loss.item(), end=end)
        if flag != False:
            self.index = soft_degree_sampling_prob(self.graph, embedding)
            self.updated_adj = graph_refine(embedding, self.adj, rate = self.hyperpm.ratio, ncaps = self.hyperpm.ncaps, node_index = self.index)
            self.neib_sampler.update(self.updated_adj)
            
        return loss.item()

    def print_trn_acc(self):
        print('trn-', end='')
        trn_acc, _ = self._print_acc(self.trn_idx, end=' val-')
        val_acc, _ = self._print_acc(self.val_idx)
        return trn_acc, val_acc

    def print_tst_acc(self):
        print('tst-', end='')
        tst_acc, hidden = self._print_acc(self.tst_idx)
        #np.save('{}_emb.npy'.format(self.hyperpm.datname), hidden.cpu().numpy())
        print('--------------', self.model.param)
        return tst_acc

    def _print_acc(self, eval_idx, end='\n'):
        self.model.eval()
        prob, _, hidden = self.model(self.feat, self.neib_sampler.nb_all)
        prob = prob[eval_idx]
        targ = self.targ[eval_idx]
        pred = prob.max(1)[1].type_as(targ)
        acc = pred.eq(targ).double().sum() / len(targ)
        acc = acc.item()
        print('acc: %.4f' % acc, end=end)
        return acc, hidden

    def visualize(self, sav_prefix):
        g = self.graph
        n = g.number_of_nodes()
        self.model.eval()
        prob = self.model(self.feat, self.neib_sampler.nb_all).cpu()
        targ = self.targ.cpu()
        acc = prob.max(1)[1].type_as(targ).eq(targ).numpy()
        acc = [('.' if c else '?') for c in acc.astype(dtype=np.bool)]
        sets = np.zeros(n, dtype=np.float32)
        sets[self.trn_idx.cpu()] = 0
        sets[self.val_idx.cpu()] = 1
        sets[self.tst_idx.cpu()] = 2
        pos_gml = sav_prefix + '.gml'
        visualize_as_gdf(g, sav_prefix, list(range(n)), targ, pos_gml)
        visualize_as_gdf(g, sav_prefix + '_set', acc, sets, pos_gml)
        visualize_as_gdf(g, sav_prefix + '_trg', acc, targ, pos_gml)
