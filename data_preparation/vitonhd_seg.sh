#!/bin/bash

## The first parameter is dataset_list, the second parameter is train/test, and the third parameter is paid/unpaired
# example:
# bash vitonhd_warped.sh train_pairs_1018new.txt train unpaired

set +x
set -e

# Splitting dataset txt name
dataset_list=$1
# train or test
dataset_mode=$2
# paired or unpaired
paired=$3

# Dataset storage location (modified according to local directory)
dataset_dir="/home/ock/aigc/vition-HD/"
# The location where the file is saved (modified according to the local directory)
save_dir="./result_seg/"
# Warped mask location (modified according to local directory)
warp_cloth_list="/home/ock/aigc/GP-VTON-main/sample/viton_hd"

python3 ./vitonhd_seg.py --dataset_dir $dataset_dir  --save_dir $save_dir --warp_cloth_list $warp_cloth_list \
 --dataset_list $dataset_list --dataset_mode $dataset_mode --paired $paired

