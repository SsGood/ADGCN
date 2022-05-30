import numpy as np
import torch
import torch.nn.functional as fn
import matplotlib
import networkx as nx
import scipy.sparse as sp
from scipy.spatial import distance


def distance_kernel(x, x_neighbor, kernel = 'sin'):
    x = x.unsqueeze(0)
    dist = distance.cdist(x.detach().cpu(), x_neighbor.detach().cpu(), metric = 'euclidean')
    if kernel == 'sin':
        dist_sin = np.sin(dist)
        dist = dist_sin ** 2
    elif kernel == 'RBF':
        sigma = np.mean(dist)
        gauss_dist = np.exp(- dist ** 2 / (2 * sigma ** 2))
        dist = 1 - gauss_dist
    return dist

def graph_refine(feat, adj, ncaps, rate, node_index, nbsz = 50, kernel = 'cosine'):
    n, d = feat.size()
    adj = adj.astype('float')
    disen_feat = feat.view(n, ncaps, d // ncaps)
    
    if kernel == 'cosine':
        disen_feat = fn.normalize(disen_feat, dim = 2)
        temp = disen_feat[node_index].unsqueeze(dim=1).cpu().detach() * disen_feat.cpu().detach()
        temp = temp.sum(dim=-1).max(dim = -1).values
        index = temp.argsort(descending = True)[:, : nbsz]
        temp_np = temp.numpy()
        for i in range(len(node_index)):
            adj[node_index[i]][index[i]] = (1 - rate) * adj[node_index[i]][index[i]] + rate * temp_np[i][index[i]]
            #print(adj[node_index[i]][index[i]])
    return adj

def soft_degree_sampling_prob(graph, feat, rate = 0.05):
    n = graph.number_of_nodes()
    node_index = list(range(n))
    node_degree = [graph.degree(index) for index in node_index]
    total_degree = sum(node_degree)
    degree = np.array(node_degree)/total_degree
    sample_node = []
    prob = np.where(degree>(1/total_degree), degree, 0)
    for i in range(int(rate*n)):
        if sum(prob) == 0.0:
            break
        prob = prob/sum(prob)
        temp = np.argmax(prob)
        sample_node.append(temp)
        neighbor_temp = list(graph.neighbors(temp))
        neighbor_temp.append(temp)
        prob[neighbor_temp] = prob[neighbor_temp] * distance_kernel(feat[temp], feat[neighbor_temp]).squeeze()
    return sample_node

def AdversarialSampler(x, num, k, dev):
    '''
    return fake sample and return the channel label
    '''
    index_k = np.random.choice(k, num)
    index_n = np.random.choice(x.size(0), num)
    x_temp = x[index_n]
    x_sample = []
    
    for i,j in enumerate(index_k):
        x_sample.append(x_temp[i,j,:])
        
    x_sample = torch.stack(x_sample, dim = 0).to(dev)
    index_k = torch.from_numpy(index_k).to(dev)
    return x_sample, index_k

def GANloss(d_real, d_fake, dev):
    '''GANloss for G and D'''
    
    loss_type = 'BCE_loss'
    if loss_type == 'BCE_loss':
        real_label = torch.full((d_real.size(0),1), 1.0, device = dev)
        fake_label = torch.full((d_real.size(0),1), 0.0, device = dev)
        d_loss_real = fn.binary_cross_entropy_with_logits(d_real, real_label)
        d_loss_fake = fn.binary_cross_entropy_with_logits(d_fake, fake_label)
        g_loss_fake = fn.binary_cross_entropy_with_logits(d_fake, real_label)
        d_loss = d_loss_real + d_loss_fake
        g_loss = g_loss_fake
    
    elif loss_type == 'hinge_loss':
        d_loss_real = torch.nn.ReLU()(1.0 - d_real).mean()
        d_loss_fake = torch.nn.ReLU()(1.0 + d_fake).mean()
        d_loss = d_loss_real + d_loss_fake
        g_loss = - d_fake.mean()
        
    return d_loss, g_loss


def clsLoss(prob, label):
    '''classification loss'''
    loss = fn.cross_entropy(prob, label)
    return loss


def thsprs_from_spsprs(x):
    x = x.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((x.row, x.col)).astype(np.int32)).long()
    val = torch.from_numpy(x.data)
    return torch.sparse.FloatTensor(idx, val, torch.Size(x.shape))


# noinspection PyUnresolvedReferences
def real2col(x):
    assert 0.0 <= x <= 1.0
    r, g, b, a = matplotlib.cm.gist_ncar(x)
    return '%d,%d,%d' % (r * 255, g * 255, b * 255)

def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor(
            torch.LongTensor(indices),
            torch.FloatTensor(coo.data),
            coo.shape)


def matrix_to_torch(X):
    if sp.issparse(X):
        return sparse_matrix_to_torch(X)
    else:
        return torch.FloatTensor(X)
    



def visualize_as_gdf(g, savfile, label, color, pos_gml=None):
    assert isinstance(g, nx.Graph)
    n = g.number_of_nodes()
    if not savfile.endswith('.gdf'):
        savfile += '.gdf'
    assert len(label) == n
    color = np.asarray(color, dtype=np.float32).copy()
    color = (color - color.min()) / (color.max() - color.min() + 1e-6)
    assert color.shape == (n,)
    if isinstance(pos_gml, str) and os.path.isfile(pos_gml):
        layout_g = nx.read_gml(pos_gml)
        layout_g = dict(layout_g.nodes)
        pos = np.zeros((n, 2), dtype=np.float64)
        for t in range(n):
            pos[t] = (layout_g[str(t)]['graphics']['x'],
                      layout_g[str(t)]['graphics']['y'])
        scale = 1
    else:
        pos = nx.random_layout(g)
        scale = 1000
    with open(savfile, 'w') as fout:
        fout.write('nodedef>name VARCHAR,label VARCHAR,'
                   'x DOUBLE,y DOUBLE,color VARCHAR\n')
        for t in range(n):
            fout.write("%d,%s,%f,%f,'%s'\n" %
                       (t, label[t], pos[t][0] * scale, pos[t][1] * scale,
                        real2col(color[t])))
        fout.write('edgedef>node1 VARCHAR,node2 VARCHAR\n')
        for (u, v) in g.edges():
            fout.write('%d,%d\n' % (u, v))