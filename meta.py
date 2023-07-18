#!/usr/bin/env python3
#
# This is the script I use to tune the hyper-parameters automatically.
#
import subprocess

import hyperopt

min_y = 0
min_c = None


def trial(hyperpm):
    global min_y, min_c
    # Plz set nbsz manually. Maybe a larger value if you have a large memory.
    cmd = 'python main.py'
    cmd = 'CUDA_VISIBLE_DEVICES=5 ' + cmd
    for k in hyperpm:
        v = hyperpm[k]
        cmd += ' --' + k
        if int(v) == v:
            cmd += ' %d' % int(v)
        else:
            cmd += ' %g' % float('%.1e' % float(v))
    try:
        val, tst = eval(subprocess.check_output(cmd, shell=True))
    except subprocess.CalledProcessError:
        print('...')
        return {'loss': 0, 'status': hyperopt.STATUS_FAIL}
    print('val=%5.2f%% tst=%5.2f%% @ %s' % (val * 100, tst * 100, cmd))
    score = -val
    if score < min_y:
        min_y, min_c = score, cmd
    return {'loss': score, 'status': hyperopt.STATUS_OK}

parser = argparse.ArgumentParser()
parser.add_argument('--datname', type=str, default='cora',
                    help='pubmed, cora, citeseer')
parser.add_argument('--nbsz', type=str, default='20',
                    help='the numbers of neighborhood ')

args = parser.parse_args()

space = {'lr': hyperopt.hp.loguniform('lr', -8, 0),
         'reg': hyperopt.hp.loguniform('reg', -10, 0),
         'nlayer': hyperopt.hp.quniform('nlayer', 1, 6, 1),
         'ncaps': 5,
         'nhidden': hyperopt.hp.quniform('nhidden', 2, 32, 2),
         'dropout': hyperopt.hp.uniform('dropout', 0, 1),
         'routit': hyperopt.hp.quniform('routit', 2, 8, 1),
         'datname': args.datname,
         'nbsz': args.nbsz}
hyperopt.fmin(trial, space, algo=hyperopt.tpe.suggest, max_evals=1000)
print('>>>>>>>>>> val=%5.2f%% @ %s' % (-min_y * 100, min_c))
