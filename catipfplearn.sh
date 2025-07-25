#!/bin/bash

twenty_datasets=("nltcs" "kdd" "plants" "baudio" "jester" \
                "bnetflix" "accidents" "tretail" "pumsb_star" "dna" \
                "kosarek" "msweb" "book" "tmovie" "cwebkb" "cr52" \
                "c20ng" "bbc" "ad" "msnbc")

my_datasets=("imgn")

for dataset in "${my_datasets[@]}"
do
    log_file_name="logs/${dataset}.txt"
    output_model_file="models/${dataset}.pt"
    if [ -e $log_file_name ]
    then
        echo "$log_file_name exists"
    else
        echo "Training on ${dataset}"
        CUDA_VISIBLE_DEVICES=0 python3 catipfp_train.py --dataset_path datasets/ \
                --dataset $dataset --device cuda:0 --model MoAT --max_epoch 100 \
                --batch_size 256 --lr 0.01 \
                --log_file $log_file_name --output_model_file $output_model_file
    fi
done
