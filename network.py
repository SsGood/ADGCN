import numpy as np
import scipy.sparse as spsprs
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from utils import *

class Discriminator(nn.Module):
    def __init__(self, nfeatures, ncaps):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(nfeatures, nfeatures//2)
        #self.linear2 = nn.Linear(nfeatures//2, nfeatures//4)
        self.cls1 = nn.Linear(nfeatures//2, 1)
        self.cls2 = nn.Linear(nfeatures//2, ncaps)
    def forward(self,x):
        x = fn.relu(self.linear(x))
        #x = fn.relu(self.linear2(x))
        logits = self.cls1(x)
        y_ = self.cls2(x)
        return logits, y_

class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):  # *nn.Linear* does not accept sparse *x*.
        return torch.mm(x, self.weight) + self.bias
    
    
class RoutingLayer(nn.Module):
    def __init__(self, dim, num_caps):#, first_layer = False):
        super(RoutingLayer, self).__init__()
        assert dim % num_caps == 0
        self.d, self.k = dim, num_caps
        self._cache_zero_d = torch.zeros(1, self.d)
        self._cache_zero_k = torch.zeros(1, self.k)

    def forward(self, x, neighbors, max_iter, param):
        dev = x.device
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
            self._cache_zero_k = self._cache_zero_k.to(dev)
            
        n, m = x.size(0), neighbors.size(1)# // x.size(0)
        d, k, delta_d = self.d, self.k, self.d // self.k
        x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = torch.cat([x, self._cache_zero_d], dim=0)
        z = z[neighbors].view(n, m, k, delta_d)
        u = None
        u_before = x.view(n, k, delta_d)
        for clus_iter in range(max_iter):
            if u is None:
                p = self._cache_zero_k.expand(n * m, k).view(n, m, k)
            else:
                p = torch.sum(z * u.view(n, 1, k, delta_d), dim=3)
            p1 = fn.softmax(p, dim=1)
            u1 = torch.sum(z * p1.view(n, m, k, 1), dim=1)
            
            p2 = fn.softmax(p, dim=2)
            u2 = torch.sum(z * p2.view(n, m, k, 1), dim=1)
            
            u = param * u1 + (1-param) * u2 
            
            u += u_before
            if clus_iter < max_iter - 1:
                u = fn.normalize(u, dim=2)
        return u.view(n, d)

    
class CapsuleNet(nn.Module):  # CapsuleNet = DisenGCN
    def __init__(self, nfeat, nclass, hyperpm):
        super(CapsuleNet, self).__init__()
        self.nhidden = hyperpm.nhidden
        self.ncaps  = hyperpm.ncaps
        rep_dim = self.nhidden * self.ncaps
        self.pca = SparseInputLinear(nfeat, rep_dim)
        self.param = nn.Parameter(torch.ones(1))
        conv_ls = []
        for i in range(hyperpm.nlayer):
            conv = RoutingLayer(rep_dim, self.ncaps)
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls
        self.mlp = nn.Linear(rep_dim, nclass)
        self.dropout = hyperpm.dropout
        self.routit = hyperpm.routit
        self.discriminator = Discriminator(self.nhidden, self.ncaps)
        

    def _dropout(self, x):
        return fn.dropout(x, self.dropout, training=self.training)

    def forward(self, x, nb):
        dev = x.device
        x = fn.relu(self.pca(x))
        x_initial = x
        param = torch.sigmoid(self.param)
        for conv in self.conv_ls:
            x = self._dropout(fn.relu(conv(x, nb, self.routit, param)))
        hidden = x.detach()
        logit = self.mlp(x)
        
        n,m = x_initial.size(0), nb.size(1)
        k = self.ncaps
        x_initial_disen = x_initial.view(n, self.ncaps, self.nhidden)
        x_disen = x.clone().detach().view(n, self.ncaps, self.nhidden)
        
        fake_sample, fake_label = AdversarialSampler(x_initial_disen, 10 * m, k, dev)
        real_sample, real_label = AdversarialSampler(x_disen, 10 * m, k, dev)
        fake_sample_D = fake_sample.clone().detach()
        real_sample_D = real_sample.clone().detach()
        
        total_loss_G = self.adv_calculate(fake_sample, real_sample, real_label, mode='G')
        total_loss_D = self.adv_calculate(fake_sample_D, real_sample_D, fake_label, mode='G')
        
        return fn.log_softmax(logit, dim=1), [total_loss_G, total_loss_D], hidden
    
    def adv_calculate(self, fake_sample, real_sample, label, mode):
        dev = fake_sample.device
        d_fake, prob_fake = self.discriminator(fake_sample)
        d_real, prob_real = self.discriminator(real_sample)
        _, adv_loss = GANloss(d_real, d_fake, dev)
        if mode == 'G':
            cls_loss = clsLoss(prob_real, label)
        elif mode == 'D':
            cls_loss = clsLoss(prob_fake, label)
        else:
            raise AssertionError("Plz specify the type of mode")
        total_loss  = adv_loss + cls_loss
#         total_loss  = cls_loss
        return total_loss
    
# class RoutingLayer(nn.Module):
#     def __init__(self, dim, num_caps):#, first_layer = False):
#         super(RoutingLayer, self).__init__()
#         assert dim % num_caps == 0
#         self.d, self.k = dim, num_caps
#         self._cache_zero_d = torch.zeros(1, self.d)
#         self._cache_zero_k = torch.zeros(1, self.k)
        
# #         if first_layer:
# #             fc = []
# #             for i in range(num_caps):
# #                 mlp = nn.Linear(dim // num_caps, dim // num_caps)
# #                 self.add_module('linear_%d'% i, mlp)
# #                 fc.append(mlp)
# #             self.fc = fc
                

#     def forward(self, x, neighbors, max_iter, param):
#         dev = x.device
#         if self._cache_zero_d.device != dev:
#             self._cache_zero_d = self._cache_zero_d.to(dev)
#             self._cache_zero_k = self._cache_zero_k.to(dev)
            
            
# #         n, m = x.size(0), neighbors.size(1)# // x.size(0)
# #         d, k, delta_d = self.d, self.k, self.d // self.k
            
# #         if hasattr(self, 'fc'):
# #             xTemp = []
# #             x_disen = fn.normalize(x.view(n, k, delta_d), dim = 2)
# #             for i, gcn in enumerate(self.fc):
# #                 temp = fn.relu(gcn(x_disen[:,i,:]))
# #                 xTemp.append(temp)
# #                 x = torch.cat(xTemp, dim=-1)
                
                
            
#         n, m = x.size(0), neighbors.size(1)# // x.size(0)
#         d, k, delta_d = self.d, self.k, self.d // self.k
#         x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
#         z = torch.cat([x, self._cache_zero_d], dim=0)
#         z = z[neighbors].view(n, m, k, delta_d)
#         u = None
#         u_before = x.view(n, k, delta_d)
#         for clus_iter in range(max_iter):
#             if u is None:
#                 p = self._cache_zero_k.expand(n * m, k).view(n, m, k)
#             else:
#                 p = torch.sum(z * u.view(n, 1, k, delta_d), dim=3)
#             p1 = fn.softmax(p, dim=1)
#             u1 = torch.sum(z * p1.view(n, m, k, 1), dim=1)
            
#             p2 = fn.softmax(p, dim=2)
#             u2 = torch.sum(z * p2.view(n, m, k, 1), dim=1)
            
#             u = param * u1 + (1-param) * u2 
            
#             u += u_before
#             if clus_iter < max_iter - 1:
#                 u = fn.normalize(u, dim=2)
#         return u.view(n, d)