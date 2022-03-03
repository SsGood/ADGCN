import subprocess



macro_f1s = []
micro_f1s = []
f1s = []
f1 = 0
for _ in [0, 0.2, 0.4, 0.6, 0.8]:  # Plz use a larger sample.
    cmd = 'CUDA_VISIBLE_DEVICES=2 python main.py --datname cora --dropout 0.092 --flag 1 --lr 0.011 --nbsz 20 --ncaps 5 --nhidden 16 --nlayer 6 --ratio 0.2 --reg 0.0089 --routit 4'
    cmd += ' --ratio' + ' %f'% _
    print(cmd)
    output = subprocess.check_output(cmd, shell=True)
    #print(output)
    #print(eval(output))
    (val, tst) = eval(output)
    macro_f1s.append(val)
    micro_f1s.append(tst)
    f1s.append(f1)
    f= open("cora_change_ratio.txt","a+")
    f.write('>>>>>>>>> min val, tst=%5.2f%% ,%5.2f%% ,%5.2f%% --ratio %f \n' % (val * 100, tst * 100, f1, _))
    
    cmd = 'CUDA_VISIBLE_DEVICES=2 python main.py --datname citeseer --dropout 0.13 --flag 1 --lr 0.0031 --nbsz 20 --ncaps 5 --nhidden 30 --nlayer 10 --ratio 0.2 --reg 0.0053 --routit 3'
    cmd += ' --ratio' + ' %f'% _
    print(cmd)
    output = subprocess.check_output(cmd, shell=True)
    #print(output)
    #print(eval(output))
    (val, tst) = eval(output)
    macro_f1s.append(val)
    micro_f1s.append(tst)
    f1s.append(f1)
    f= open("citeseer_change_ratio.txt","a+")
    f.write('>>>>>>>>> min val, tst=%5.2f%% ,%5.2f%% ,%5.2f%% --ratio %f \n' % (val * 100, tst * 100, f1, _))
    
    cmd = 'CUDA_VISIBLE_DEVICES=2 python main.py --datname pubmed --dropout 0.15 --flag 1 --lr 0.0027 --nbsz 20 --ncaps 5 --nhidden 28 --nlayer 4 --ratio 0.2 --reg 0.0063 --routit 3'
    cmd += ' --ratio' + ' %f'% _
    print(cmd)
    output = subprocess.check_output(cmd, shell=True)
    #print(output)
    #print(eval(output))
    (val, tst) = eval(output)
    macro_f1s.append(val)
    micro_f1s.append(tst)
    f1s.append(f1)
    f= open("pubmed_change_ratio.txt","a+")
    f.write('>>>>>>>>> min val, tst=%5.2f%% ,%5.2f%% ,%5.2f%% --ratio %f \n' % (val * 100, tst * 100, f1, _))

