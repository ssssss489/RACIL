#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

seeds=(0 1 2 3 4)
datasets=(cifar100_10_ miniimageNet_10_)
models=(ER, UCIR)

# for seed in ${seeds[*]}
# do
#  python3 main.py --result_path result_2/ --dataset cifar100_10_$seed --model ER --n_epochs 6 --lr 0.01 --n_memories 2000 --bn_type bn --regular_type None
#  python3 main.py --result_path result_2/ --dataset cifar100_10_$seed --model ER --n_epochs 6 --lr 0.01 --n_memories 2000 --bn_type bn --regular_type decoder
#  python3 main.py --result_path result_2/ --dataset cifar100_10_$seed --model ER --n_epochs 6 --lr 0.01 --n_memories 2000 --bn_type nbn --regular_type None
#  python3 main.py --result_path result_2/ --dataset cifar100_10_$seed --model ER --n_epochs 6 --lr 0.01 --n_memories 2000 --bn_type nbn --regular_type decoder
# done

for seed in ${seeds[*]}
do
#  python3 main.py --result_path result_2/ --dataset cifar100_10_$seed --model UCIR --n_epochs 6 --lr 0.01 --n_memories 2000 --bn_type bn --regular_type None
  python3 main.py --result_path result_2/ --dataset cifar100_10_$seed --model UCIR --n_epochs 6 --lr 0.01 --n_memories 2000 --bn_type bn --regular_type decoder
#  python3 main.py --result_path result_2/ --dataset cifar100_10_$seed --model UCIR --n_epochs 6 --lr 0.01 --n_memories 2000 --bn_type nbn --regular_type None
  python3 main.py --result_path result_2/ --dataset cifar100_10_$seed --model UCIR --n_epochs 6 --lr 0.01 --n_memories 2000 --bn_type nbn --regular_type decoder
done

#for seed in ${seeds[*]}
#do
#  python3 main.py --result_path result_2/ --dataset miniimageNet64_10_$seed --model ER --n_epochs 6 --lr 0.01 --batch_size 100 --n_memories 2000 --bn_type bn --regular_type None
#  python3 main.py --result_path result_2/ --dataset miniimageNet64_10_$seed --model ER --n_epochs 6 --lr 0.01 --batch_size 100 --n_memories 2000 --bn_type bn --regular_type decoder
#  python3 main.py --result_path result_2/ --dataset miniimageNet64_10_$seed --model ER --n_epochs 6 --lr 0.01 --batch_size 100 --n_memories 2000 --bn_type nbn --regular_type None
#  python3 main.py --result_path result_2/ --dataset miniimageNet64_10_$seed --model ER --n_epochs 6 --lr 0.01 --batch_size 100 --n_memories 2000 --bn_type nbn --regular_type decoder
#done
