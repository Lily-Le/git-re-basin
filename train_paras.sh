#!/bin/bash    
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
for seed in 0 42 579 1000
do python src/cifar10_resnet20_train.py --dataset cifar10-merged --seed $seed --gpu 3080
done