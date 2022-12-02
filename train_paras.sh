#!/bin/bash    
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
for seed in 0 42 579 1000 654 234 12 87 645 24
do 
    python src/mnist_mlp_train.py --dataset "mnist" --seed $seed 
    for noise_type in "glass_blur" 
    do
        python src/mnist_mlp_train.py --dataset "mnist-corrupted" --seed $seed --noise_type $noise_type
        python src/mnist_mlp_train.py --dataset "mnist-merged" --seed $seed --noise_type $noise_type "identity"

    done
   
done