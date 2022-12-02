#!/bin/bash  
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7  
model_a="mnist-mlp-weights-new:seed0-gpu3080"
seed0=0
name='mnistm-to-mnist'
for seed in 0 42 579 1000 
do 
model_b="mnist-merged-mlp-weights-new:seed"$seed"-gpu3080"
python src/align_mnist.py --model-a $model_a --model-b $model_b --seed $seed0 --name $name
done

