python main.py --datname cora        --dropout 0.17   --flag 1 --lr 0.025  --nbsz 20 --ncaps 5 --nhidden 32 --nlayer 10 --ratio 0   --reg 0.0052  --routit 6
python main.py --datname citeseer    --dropout 0.055  --flag 1 --lr 0.0067 --nbsz 20 --ncaps 5 --nhidden 26 --nlayer 9  --ratio 0.6 --reg 0.014   --routit 9
python main.py --datname pubmed      --dropout 0.17   --flag 1 --lr 0.012  --nbsz 20 --ncaps 5 --nhidden 24 --nlayer 9  --ratio 0   --reg 0.015   --routit 7
python main.py --datname computers   --dropout 0.02   --flag 1 --lr 0.0031 --nbsz 20 --ncaps 5 --nhidden 32 --nlayer 2  --ratio 0.6 --reg 0.0039  --routit 9
python main.py --datname photo       --dropout 0.023  --flag 1 --lr 0.007  --nbsz 20 --ncaps 5 --nhidden 32 --nlayer 2  --ratio 0.1 --reg 0.0024  --routit 4
python main.py --datname ms_academic --dropout 0.027  --flag 1 --lr 0.0023 --nbsz 20 --ncaps 5 --nhidden 30 --nlayer 2  --ratio 0.8 --reg 0.00086 --routit 8