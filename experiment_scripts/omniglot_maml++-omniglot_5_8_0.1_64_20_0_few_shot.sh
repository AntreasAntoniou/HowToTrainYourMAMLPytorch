#!/bin/sh

export GPU_ID=$1

echo $GPU_ID

cd ..
export DATASET_DIR="datasets/"
export CUDA_VISIBLE_DEVICES=$GPU_ID
# Activate the relevant virtual environment:
python src/run.py --name_of_args_json_file experiment_config/omniglot_maml++-omniglot_5_8_0.1_64_20_0.json --gpu_to_use $GPU_ID