#%% Read pkl file
#%% export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
import pickle
import argparse
import pickle
from pathlib import Path
import os
import jax.numpy as jnp
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
# from cifar10_resnet20_weight_matching import load_model
#%%
# model = ResNet(blocks_per_group=BLOCKS_PER_GROUP["resnet20"],
#                    num_classes=10,
#                    width_multiplier=1)
# stuff = make_stuff(model)

# def load_model(filepath):
#       with open(filepath, "rb") as fh:
#         return from_bytes(
#             model.init(random.PRNGKey(0), jnp.zeros((1, 32, 32, 3)))["params"], fh.read())
#%%
parser = argparse.ArgumentParser()
parser.add_argument("--model-a", type=str)
parser.add_argument("--model-b", type=str)
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--name", type=str,required=True)
args = parser.parse_args()   
load_epoch=99

#%% Analyze weights of models trained with the same dataset 
# seed0=0
# seed_=42
# prefix_='mnist-merged-mlp-weights-new'
# name='mnist-mnist'
# # dataset='cifar10'

# seed=0


# model_a_name=prefix_+':'+f'seed{seed0}-gpu3080'
# model_b_name=prefix_+':'+f'seed{seed_}-gpu3080'

#%%

with wandb.init(
        project="git-re-basin",
        entity="lily-le",
        name=f"{args.name}-match",
        tags=["mnist", "mlp", "weight-matching"],
        job_type="analysis-matched-weight",
        # job_type="debug",
) as wandb_run:
    config = wandb.config
    # config.load_epoch = load_epoch
    # config.dataset=args.dataset
    # config.model_a_name = args.model_a
    # config.model_b_name = args.model_b
    # config.seed=args.seed
    config.load_epoch = load_epoch
    # config.dataset=dataset
    config.model_a_name = args.model_a
    config.model_b_name = args.model_b
    config.seed=args.seed
    config.name=args.name

    model = MLPModel()
    stuff = make_stuff(model)

    def load_model(filepath):
        with open(filepath, "rb") as fh:
            return from_bytes(
                model.init(random.PRNGKey(0), jnp.zeros((1, 28, 28, 1)))["params"], fh.read())

    filename = f"checkpoint{load_epoch}"
    model_a = load_model(
        Path(
            wandb_run.use_artifact(f"{config.model_a_name}").get_path(
                filename).download()))
    
    
    model_b=load_model(
    Path(
        wandb_run.use_artifact(f"{config.model_b_name}").get_path(
            filename).download()))

    # models_to_align.append(model_b)
    permutation_spec =mlp_permutation_spec(3)#resnet20_permutation_spec()# 
    final_permutation_b = weight_matching(random.PRNGKey(config.seed), permutation_spec,
                                    flatten_params(model_a), flatten_params(model_b))
    model_b_clever = unflatten_params(
        apply_permutation(permutation_spec, final_permutation_b, flatten_params(model_b)))
    
    def create_dirs(prefix_name):
        if not os.path.exists(prefix_name):
            os.makedirs(prefix_name)
    
    prefix_name=f'output/{config.name}-{config.model_a_name}-matched/'.replace(':','')
    diff_path=os.path.join(prefix_name,'diff_results')
    model_save_path=os.path.join(prefix_name,'matched_models')
    # prefix_name2=f'output/{config.model_b_name}-clever/'.replace(':','')
    create_dirs(diff_path)
    create_dirs(model_save_path)
    # create_dirs(prefix_name2)
    result={}
    for i in model_a:
        result[i]={}
        ly_a=model_a[i]
        ly_b=model_b_clever[i]
        keys=ly_a.keys()
        for k in ly_a.keys():
            result[i][k]={}

            k_a=ly_a[k]
            k_b=np.array(ly_b[k])
            
            sub=k_b-k_a
            # num_paras=sub.shape()
            sub_mean=np.sum(sub)/sub.size
            MSE_mean=np.sum(sub*sub)/sub.size
            # result[i][k]['sub']=(sub).astype(np.float64).tolist()
            result[i][k]['sub_mean']=float(sub_mean)
            result[i][k]['MSE_mean']=float(MSE_mean)
    b = json.dumps(result)
    f2 = open(os.path.join(diff_path,f'{config.model_b_name}.json'.replace(':','')), 'w')
    f2.write(b)
    f2.close()
    f2 = open(os.path.join(model_save_path,f'{config.model_b_name}.pkl'.replace(':','')), 'wb')
    pickle.dump(model_b_clever,f2)
    f2.close()
    f2 = open(os.path.join(prefix_name,f'{config.model_a_name}.pkl'.replace(':','')), 'wb')
    pickle.dump(model_a,f2)
    f2.close()
    # np.save(os.path.join(prefix_name,f'{i}.npy'),X_tsne)
    # with open(filename, mode="wb") as f:
    #     f.write(flax.serialization.to_bytes(train_state.params))

    
    artifact = wandb.Artifact(f"{config.name}_mlp_weight_matching",
                        type="matched-model",
                        metadata={
                            "dataset": config.name,
                            "matched":config.model_a_name,
                            "model": "mlp",
                            "model_a":config.model_a_name,
                            "model_b":config.model_b_name,
                            "analysis": "weight-matching"
                        })
    with artifact.new_file(f"clever_{config.model_b_name}.pkl".replace("mlp-weights-new:",""), mode="wb") as f:
        pickle.dump(model_b_clever, f)
    # with artifact.new_file(f"diff.json", mode="w") as f2:
    #     json.dump(result, f2)  
    wandb_run.log_artifact(artifact,aliases=[f'{config.model_a_name}+{config.model_b_name}'.replace("mlp-weights-new:","")])
    
    # artifact.add_file(filename)

    # This will be a no-op when config.test is enabled anyhow, since wandb will
    # be initialized with mode="disabled".
        # wandb_run.log_artifact(artifact)

    # model_c_clever = unflatten_params(
    #         apply_permutation(permutation_spec, final_permutation_c, flatten_params(model_c)))
    
    # wandb_run.log({
    #       "epoch": epoch,
    #       "train_loss": train_loss,
    #       "test_loss": test_loss,
    #       "train_accuracy": train_accuracy,
    #       "test_accuracy": test_accuracy,
    #   })

      # No point saving the model at all if we're running in test mode.


#%%
'''
import flax
model_a_name=prefix_+':'+f'seed{seed0}-gpu3080'
model_b_name=prefix_+':'+f'seed{seed_}-gpu3080'

# config.model_a_name = model_a_name
# config.model_b_name = model_b_name
def create_dirs(prefix_name):
    if not os.path.exists(prefix_name):
        os.makedirs(prefix_name)

prefix_name=f'output/{model_a_name}-matched/'.replace(':','')
diff_path=os.path.join(prefix_name,'diff_results')
model_save_path=os.path.join(prefix_name,'matched_models')
# prefix_name2=f'output/{config.model_b_name}-clever/'.replace(':','')
create_dirs(diff_path)
create_dirs(model_save_path)

result={}
for i in model_a:
    result[i]={}
    ly_a=model_a[i]
    ly_b=model_b_clever[i]
    keys=ly_a.keys()
    for k in ly_a.keys():
        result[i][k]={}

        k_a=ly_a[k]
        k_b=np.array(ly_b[k])
        
        sub=k_b-k_a
        sub_sum=np.sum(sub)
        MSE=np.sum(sub*sub)
        result[i][k]['sub']=(sub).astype(np.float64).tolist()
        result[i][k]['sub_sum']=float(sub_sum)
        result[i][k]['MSE']=float(MSE)

b = json.dumps(result)
f2 = open(os.path.join(diff_path,f'{model_b_name}.json'.replace(':','')), 'w')
f2.write(b)
f2.close()
f2 = open(os.path.join(model_save_path,f'{model_b_name}.pkl'.replace(':','')), 'wb')
f2.write(flax.serialization.to_bytes(pickle.dumps((model_b_clever))))
f2.close()


'''
#%%
    # prefix_name=f'output/{config.model_a_name}/'.replace(':','')
    # prefix_name2=f'output/{config.model_b_name}-clever/'.replace(':','')
    # create_dirs(prefix_name)
    # create_dirs(prefix_name2)

        # tSNE = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
        # X_tsne = tSNE.fit_transform(k_a)
        # print(f'tsne {i} finished!')
        # prefix_name=f'output/{config.model_a_name}/tsne{n_components}/'.replace(':','')
        # if not os.path.exists(prefix_name):
        #     os.makedirs(prefix_name)
        # np.save(os.path.join(prefix_name,f'{i}.npy'),X_tsne)
             #   with open(filename, mode="wb") as f:
        # #     f.write(flax.serialization.to_bytes(train_state.params))

# 将原来的数据进行聚类，斌知道是哪一类的
#%%

# #设置聚类
# n_clusters=6
# kmeans = KMeans(n_clusters=n_clusters, random_state=2018)
# for i in model_a:
#     paras=np.load(os.path.join(prefix_name,f'{i}.npy'))
#     kmeans.fit(paras)
#     pre_y = kmeans.predict(paras)
#     centers = kmeans.cluster_centers_
#     plt.scatter(paras[:, 0], paras[:, 1], paras[:,2],c=pre_y[:])
#     plt.show()

# # 建立聚类模型对象
# # 训练聚类模型

# # 预测聚类模型



#     # k_c=np.array(ly_c['kernel'])
#     # distab=cdist(k_a,k_b,metric='cityblock')
    

#     # print(dist)

# # %%
# import seaborn as sns
# sns.heatmap(data=k_a)
# # %%
# sns.heatmap(data=k_b)
# # %%

# sns.heatmap(data=k_a-k_b)
# # %%
# # %%
# np.sum(k_a-k_b)
# # %%
# # %%
# np.mean(k_b)
# # %%
# # %%
# import numpy as np
# import pandas as pd
# from sklearn import manifold
# from sklearn.cluster import KMeans
# from sklearn import metrics
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
# #%matplotlib inline

# # 数据准备
# # data = make_blobs(n_samples=2000, centers=[[1,1], [-1, -1]], cluster_std=0.7, random_state=2018)

# #%%
# # 颜色设置
# # colors = ['green', 'pink']
# # 创建画布
# plt.figure(figsize=(12,6))
# titles = ['Real', 'Predict']
# colors = ["blue","black","brown","red","yellow","green","orange","beige","turquoise","pink"]
# for j, y_ in enumerate(np.array([y, pre_y])):
#     plt.subplot(1,2, j+1)
#     plt.title(titles[j])
#     # 循环读类别
#     for i in range(n_clusters):
#         # 找到相同的索引
#         index_sets = np.where(y_ == i)
#         # 将相同类的数据划分为一个聚类子集
#         cluster = X[index_sets]
#         # 展示样本点
#         plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], marker='.')
#         if j==1:          
#         # 簇中心
#             plt.plot(centers[i][0], centers[i][1], 'o',markerfacecolor=colors[i],markeredgecolor='k', markersize=6)
# # plt.savefig('xx.png')
# plt.show()

# #%%

# #%%
# ### 模型效果指标评估 ###
# # 样本距离最近的聚类中心的总和
# inertias = kmeans.inertia_

# # 调整后的兰德指数
# adjusted_rand_s = metrics.adjusted_rand_score(y, pre_y)

# # 互信息
# mutual_info_s = metrics.mutual_info_score(y, pre_y)

# # 调整后的互信息
# adjusted_mutual_info_s = metrics.adjusted_mutual_info_score(y, pre_y)

# # 同质化得分
# homogeneity_s = metrics.homogeneity_score(y, pre_y)

# # 完整性得分
# completeness_s = metrics.completeness_score(y, pre_y)

# # V-measure得分
# v_measure_s = metrics.v_measure_score(y, pre_y)

# # 平均轮廓系数
# silhouette_s = metrics.silhouette_score(X, pre_y, metric='euclidean')

# # Calinski 和 Harabaz 得分
# calinski_harabaz_s = metrics.calinski_harabaz_score(X, pre_y)

# df_metrics = pd.DataFrame([[inertias, adjusted_rand_s,mutual_info_s, adjusted_mutual_info_s, homogeneity_s,completeness_s,v_measure_s, silhouette_s ,calinski_harabaz_s]],
#                          columns=['ine','tARI','tMI','tAMI','thomo','tcomp','tv_m','tsilh','tc&h'])

# df_metrics
