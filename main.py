import argparse
import os
import pickle
import random
import sys
import tempfile
import time

import gc
import networkx as nx
import numpy as np
import scipy.sparse as spsprs
import torch
from preprocessing import *


from network import *
from utils import *
from model import *

class RedirectStdStreams:
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr



# noinspection PyUnresolvedReferences
def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# noinspection PyUnresolvedReferences
def train_and_eval(datadir, datname, hyperpm):
    test = hyperpm.test
    test_seeds = [
        2144199730,  794209841, 2985733717, 2282690970, 1901557222,
        2009332812, 2266730407,  635625077, 3538425002,  960893189,
        497096336, 3940842554, 3594628340,  948012117, 3305901371,
        3644534211, 2297033685, 4092258879, 2590091101, 1694925034]
    val_seeds = [
        2413340114, 3258769933, 1789234713, 2222151463, 2813247115,
        1920426428, 4272044734, 2092442742, 841404887, 2188879532,
        646784207, 1633698412, 2256863076,  374355442,  289680769,
        4281139389, 4263036964,  900418539,  119332950, 1628837138]

    if test:
        seeds = test_seeds
    else:
        seeds = val_seeds
    
    best_val_acc_list = []
    best_tst_acc_list = []
    for num, seed in enumerate(seeds):
        print('current seed: ', seed, 'current number: ', num)
        if datname in ['cora','citeseer']:
            dataset = DataReader(datname, datadir, seed, test)
        else:
            dataset = DataReader_random_split(datname, datadir, seed, test)
        set_rng_seed(hyperpm.seed)
        agent = EvalHelper(dataset, hyperpm)
        #agent = EvalHelper(DataReader(datname, datadir), hyperpm)
        tm = time.time()
        best_val_acc, wait_cnt = 0.0, 0
        model_sav = tempfile.TemporaryFile()
        neib_sav = torch.zeros_like(agent.neib_sampler.nb_all, device='cpu')
        flag = False
        for t in range(hyperpm.nepoch):
            print('%3d/%d' % (t, hyperpm.nepoch), end=' ')
            _, tc = agent.run_epoch(flag, end=' ')
            _, cur_val_acc = agent.print_trn_acc()
            #print('time-consuming: ', tc)
            if cur_val_acc > best_val_acc:
                wait_cnt = 0
                best_val_acc = cur_val_acc
                model_sav.close()
                model_sav = tempfile.TemporaryFile()
                torch.save(agent.model.state_dict(), model_sav)
                neib_sav.copy_(agent.neib_sampler.nb_all)
                if cur_val_acc >= 0.7:
                    flag = hyperpm.flag
            else:
                flag = False
                wait_cnt += 1
                if wait_cnt > hyperpm.early:
                    break
        print("time: %.4f sec." % (time.time() - tm))
        model_sav.seek(0)
        agent.model.load_state_dict(torch.load(model_sav))
        agent.neib_sampler.nb_all.copy_(neib_sav)
        #agent.visualize('./{}'.format(datname))
        tst_acc, tst_f1 = agent.print_tst_acc()
        best_val_acc_list.append(best_val_acc)
        best_tst_acc_list.append(tst_acc)
        if num >= 9 and np.max(best_val_acc_list) < 0.6:
            break
    np.save('{}-{}'.format(hyperpm.datname, hyperpm.flag), np.array(best_tst_acc_list))
    return np.mean(best_val_acc_list), np.mean(best_tst_acc_list)


def main(args_str=None):
    assert float(torch.__version__[:3]) + 1e-3 >= 0.4
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data/')
    parser.add_argument('--datname', type=str, default='cora')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Insist on using CPU instead of CUDA.')
    parser.add_argument('--nepoch', type=int, default=200,
                        help='Max number of epochs to train.')
    parser.add_argument('--early', type=int, default=50,
                        help='Extra iterations before early-stopping.')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='Initial learning rate.')
    parser.add_argument('--reg', type=float, default=0.0036,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.35,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nlayer', type=int, default=5,
                        help='Number of conv layers.')
    parser.add_argument('--ncaps', type=int, default=7,
                        help='Maximum number of capsules per layer.')
    parser.add_argument('--nhidden', type=int, default=16,
                        help='Number of hidden units per capsule.')
    parser.add_argument('--routit', type=int, default=6,
                        help='Number of iterations when routing.')
    parser.add_argument('--nbsz', type=int, default=20,
                        help='Size of the sampled neighborhood.')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='Update ratio of adj')
    parser.add_argument('--seed', type=int, default=23,
                        help='random seed')
    parser.add_argument('--task', type=str, default='classification',
                        help='task')
    parser.add_argument('--n_cluster', type=int, default=7,
                        help='class number')
    parser.add_argument('--test', type=str, default=True,
                        help='val_test flag')
    parser.add_argument('--flag', type=str, default=False,
                        help='val_test flag')
    if args_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_str.split())
    with RedirectStdStreams(stdout=sys.stderr):
        if args.datname == 'cora':
            args.n_cluster = 7
        elif args.datname == 'citeseer':
            args.n_cluster = 6
        elif args.datname == 'pubmed':
            args.n_cluster = 3
        val_acc, tst_acc = train_and_eval(args.datadir, args.datname, args)
        print('val=%.2f%% tst_acc=%.2f%%' % (val_acc * 100, tst_acc * 100))
    return val_acc, tst_acc


if __name__ == '__main__':
    print(str(main()))
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
