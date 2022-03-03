#!/usr/bin/env python3
#
# This is the script I use to tune the hyper-parameters automatically.
#
import subprocess
import argparse
import hyperopt

min_y = 0
min_c = None


def trial(hyperpm):
    global min_y, min_c
    # Plz set nbsz manually. Maybe a larger value if you have a large memory.
    cmd = 'python main.py'
    #cmd = 'CUDA_VISIBLE_DEVICES=7 ' + cmd
    for k in hyperpm:
        v = hyperpm[k]
        cmd += ' --' + k
        
        if isinstance(v, str):
            cmd += ' %s' %v
        elif int(v) == v:
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
        f= open("logger-{}.txt".format(args.datname),"a+")
        f.write('>>>>>>>>>> min val now=%5.2f%% @ %s\n' % (-min_y * 100, min_c))
        f.close()
    return {'loss': score, 'status': hyperopt.STATUS_OK}


parser = argparse.ArgumentParser()
parser.add_argument('--datname', type=str, default='cora',
                    help='pubmed, cora, citeseer')
parser.add_argument('--nbsz', type=str, default='20',
                    help='the numbers of neighborhood ')
parser.add_argument('--flag', type=str, default=True,
                    help='semi-supervised task ')
args = parser.parse_args()
space = {'lr': hyperopt.hp.loguniform('lr', -8, 0),
         'reg': hyperopt.hp.loguniform('reg', -10, 0),
         'nlayer': hyperopt.hp.quniform('nlayer', 1, 10, 1),
         'ncaps': 5, #hyperopt.hp.quniform('ncaps', 3, 10, 1),
         'nhidden': hyperopt.hp.quniform('nhidden', 2, 32, 2),
         'dropout': hyperopt.hp.uniform('dropout', 0, 0.9),
         'routit': hyperopt.hp.quniform('routit', 3, 9, 1),
         'ratio': hyperopt.hp.quniform('ratio', 0, 0.5, 0.05),
        'datname': args.datname,
        'nbsz': args.nbsz,
        'flag': args.flag}
hyperopt.fmin(trial, space, algo=hyperopt.tpe.suggest, max_evals=1000)
print('>>>>>>>>>> val=%5.2f%% @ %s' % (-min_y * 100, min_c))