# ADGCN
Pytorch Implementation for paper "Adversarial Graph Disentanglement". Note: a well-organized version will be coming soon!

**2022/05/30 Update: The organized version has been released!**
## Introduction


## Requirements
* PyTorch >= 1.1.0
* python 3.6
* networkx
* scikit-learn
* scipy
* munkres

## Run from
preset hyperparameters version:
```bash
source ./pre_ADGCN.sh
```
or modifying the network hyperparameters and run
```bash
python main.py --param1 xxx --param2 xxx --param3 xxx ...
```

You can also use "meta.py" to search for the best combination of hyperparameters on each dataset:
```bash
python meta.py --datname $dataset_name
```

## Data
We provide the citation network datasets under data/. Due to space limit, please download AMZ co-purchase dataset from https://github.com/shchur/gnn-benchmark#datasets
