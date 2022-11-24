#%% Read pkl file
import pickle
import argparse
import pickle
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb
from flax.serialization import from_bytes
from jax import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mnist_mlp_train import MLPModel, load_mnist_datasets, make_stuff,load_mnistc_datasets,load_merged_datasets
from utils import ec2_get_instance_type, flatten_params, lerp, unflatten_params
from weight_matching import (apply_permutation, mlp_permutation_spec, weight_matching)
from scipy.spatial.distance import cdist

#%%
path='/home/cll/work/exp_results/permutation.pkl'
with open(path,'rb') as f:
    permu=pickle.load(f)
    f.close()


#%%
    
dataset='mnist'
load_epoch=99
model_a='mnist-mlp-weights:seed42'
model_b='mnist-mlp-weights:seed579'
model_c='mnist-corrupted-mlp-weights:seed42'
seed=0
with wandb.init(
        project="git-re-basin",
        entity="lily-le",
        name=f"{dataset}-match",
        tags=["mnist", "mlp", "weight-matching"],
        job_type="analysis",
) as wandb_run:
    model = MLPModel()
    stuff = make_stuff(model)

    def load_model(filepath):
        with open(filepath, "rb") as fh:
            return from_bytes(
                model.init(random.PRNGKey(0), jnp.zeros((1, 28, 28, 1)))["params"], fh.read())

    filename = f"checkpoint{load_epoch}"
    model_a = load_model(
        Path(
            wandb_run.use_artifact(f"{model_a}").get_path(
                filename).download()))
    model_b = load_model(
        Path(
            wandb_run.use_artifact(f"{model_b}").get_path(
                filename).download()))
    model_c = load_model(
        Path(
            wandb_run.use_artifact(f"{model_c}").get_path(
                filename).download()))

    permutation_spec = mlp_permutation_spec(3)
    final_permutation_b = weight_matching(random.PRNGKey(seed), permutation_spec,
                                        flatten_params(model_a), flatten_params(model_b))
    final_permutation_c = weight_matching(random.PRNGKey(seed), permutation_spec,
                                        flatten_params(model_a), flatten_params(model_c))
    model_b_clever = unflatten_params(
            apply_permutation(permutation_spec, final_permutation_b, flatten_params(model_b)))
    model_c_clever = unflatten_params(
            apply_permutation(permutation_spec, final_permutation_c, flatten_params(model_c)))

# %%
for i in model_a:
    ly_a=model_a[i]
    ly_b=model_b_clever[i]
    ly_c=model_c_clever[i]
    k_a=ly_a['kernel']
    k_b=np.array(ly_b['kernel'])
    k_c=np.array(ly_c['kernel'])
    # distab=cdist(k_a,k_b,metric='cityblock')
    break
    # print(dist)

# %%
import seaborn as sns
sns.heatmap(data=k_a)
# %%
sns.heatmap(data=k_b)
# %%
sns.heatmap(data=k_c)
# %%
sns.heatmap(data=k_a-k_b)
# %%
sns.heatmap(data=k_a-k_c)
# %%
np.sum(k_a-k_b)
# %%
np.sum(k_a-k_c)
# %%
np.mean(k_b)
# %%
np.mean(k_c)
# %%
