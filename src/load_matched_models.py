#%% export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
import pickle
import argparse
import pickle
from pathlib import Path
import os
# import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb
from flax.serialization import from_bytes,to_bytes
from jax import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mnist_mlp_train import MLPModel, load_mnist_datasets, make_stuff,load_mnistc_datasets,load_merged_datasets
from utils import ec2_get_instance_type, flatten_params, lerp, unflatten_params
from weight_matching import (apply_permutation, mlp_permutation_spec, weight_matching,resnet20_permutation_spec)
from scipy.spatial.distance import cdist

from cifar10_resnet20_train import BLOCKS_PER_GROUP, ResNet, make_stuff
import json

#%%
base_dirs={
    'mnist-mnist':'/home/cll/work/code/git-re-basin/output/mnist-mlp-weights-newseed0-gpu3080-matched',
    'mnist-mnistc':'/home/cll/work/code/git-re-basin/output/mnist-to-mnistc-mnist-corrupted-mlp-weights-newseed0-gpu3080-matched',
    'mnist-mnistm':'/home/cll/work/code/git-re-basin/output/mnist-to-mnistm-mnist-merged-mlp-weights-newseed0-gpu3080-matched',

    'mnistc-mnistc':'/home/cll/work/code/git-re-basin/output/mnist-corrupted-mlp-weights-newseed0-gpu3080-matched',
    'mnistc-mnist':'/home/cll/work/code/git-re-basin/output/mnistc-to-mnist-mnist-mlp-weights-newseed0-gpu3080-matched',
    'mnistc-mnistm':'/home/cll/work/code/git-re-basin/output/mnistc-to-mnistm-mnist-merged-mlp-weights-newseed0-gpu3080-matched',
    
    'mnistm-mnistm':'/home/cll/work/code/git-re-basin/output/mnist-merged-mlp-weights-newseed0-gpu3080-matched',
    'mnistm-mnistc':'/home/cll/work/code/git-re-basin/output/mnistm-to-mnistc-mnist-corrupted-mlp-weights-newseed0-gpu3080-matched',
    'mnistm-mnist':'/home/cll/work/code/git-re-basin/output/mnistm-to-mnist-mnist-mlp-weights-newseed0-gpu3080-matched',
}

# base_dir='/home/cll/work/code/git-re-basin/output/mnistc-to-mnist-mnist-mlp-weights-newseed0-gpu3080-matched'
# base_dir='/home/cll/work/code/git-re-basin/output/mnist-to-mnistc-mnist-corrupted-mlp-weights-newseed0-gpu3080-matched'
# base_dir='/home/cll/work/code/git-re-basin/output/mnist-merged-mlp-weights-newseed0-gpu3080-matched'
# base_dir='/home/cll/work/code/git-re-basin/output/mnistc-to-mnistm-mnist-merged-mlp-weights-newseed0-gpu3080-matched'
# base_dir='/home/cll/work/code/git-re-basin/output/mnistm-to-mnistc-mnist-corrupted-mlp-weights-newseed0-gpu3080-matched'
# base_dir='/home/cll/work/code/git-re-basin/output/mnist-to-mnistm-mnist-merged-mlp-weights-newseed0-gpu3080-matched'
# base_dir='/home/cll/work/code/git-re-basin/output/mnistm-to-mnist-mnist-mlp-weights-newseed0-gpu3080-matched'
base_dir=base_dirs['mnistm-mnist']
# model_name='mnist-mlp-weights-newseed42-gpu3080.pkl'
#%%
seed0=0
seeds=[0,42,579,1000]
models={}
filepath=os.path.join(base_dir,f'mnist-mlp-weights-newseed{seed0}-gpu3080.pkl')
with open(filepath, 'rb') as f:
        model0 = pickle.load(f)

for n in seeds:
    filepath=os.path.join(base_dir,'matched_models',f'mnist-merged-mlp-weights-newseed{n}-gpu3080.pkl')
    with open(filepath, 'rb') as f:
        models[n] = pickle.load(f)
# seeds.insert(0,seed0)

# %% Calculate the mean of each layer
layer_sum={}
layer_mean={}
model=model0
for i in model:
    # print(i)
    layer_sum[i]={}
    layer_mean[i]={}
    ly=model[i]
    
    for k in ly.keys():
        layer_sum[i][k]=np.float64(model[i][k].copy())
    for n in seeds:
        model_=models[n]
        ly=model_[i]
        keys=ly.keys()
        for k in ly.keys():
            layer_sum[i][k]=layer_sum[i][k]+np.float64(model_[i][k])
    for k in layer_sum[i].keys():
        layer_mean[i][k]=(layer_sum[i][k]/(len(seeds)+1)).tolist()
        # k_a=ly_a[k]
        # k_b=np.array(ly_b[k])
        
        # sub=k_b-k_a
        # sub_sum=np.sum(sub)
        # MSE=np.sum(sub*sub)
        # result[i][k]['sub']=(sub).astype(np.float64).tolist()
        # result[i][k]['sub_sum']=float(sub_sum)
        # result[i][k]['MSE']=float(MSE)

b = json.dumps(layer_mean)
f2 = open(os.path.join(base_dir,'layer_mean.json'), 'w')
f2.write(b)
# %%

result={}

for i in model:
    result[i]={}
    ly_a=model[i]
    keys=ly_a.keys()
        
    for k in ly_a.keys():
        result[i][k]={}
        k_a=ly_a[k].astype(np.float64)
        k_b=np.array(layer_mean[i][k]).astype(np.float64)  
        sub=k_a-k_b
        sub_mean=np.sum(sub)/sub.size
        MSE_mean=np.sum(sub*sub)/sub.size
        # result[i][k]['sub']=(sub).astype(np.float64).tolist()
        result[i][k]['sub_mean']=float(sub_mean)
        result[i][k]['MSE_mean']=float(MSE_mean)
    for n in seeds:
        model_a=models[n]
        ly_a=model_a[i]
        keys=ly_a.keys()
        for k in ly_a.keys():
            
            k_a=ly_a[k].astype(np.float64)
            k_b=np.array(layer_mean[i][k]).astype(np.float64)
            sub=k_a-k_b
            sub_mean=np.sum(sub)/sub.size
            MSE_mean=np.sum(sub*sub)/sub.size
            # result[i][k]['sub']=(sub).astype(np.float64).tolist()
            result[i][k]['sub_mean']=result[i][k]['sub_mean']+float(sub_mean)
            result[i][k]['MSE_mean']=result[i][k]['MSE_mean']+float(MSE_mean)
    for k in result[i].keys():
        result[i][k]['sub_mean']=(result[i][k]['sub_mean']/(len(seeds)+1))
        result[i][k]['MSE_mean']=(result[i][k]['MSE_mean']/(len(seeds)+1))

# %%
b = json.dumps(result)
f2 = open(os.path.join(base_dir,'result.json'), 'w')
f2.write(b)
f2.close()

# %%

# %%
