#!/bin/sh
cd ..
export DATASET_DIR="../datasets/"
# Activate the relevant virtual environment:

python train_maml_system.py --name_of_args_json_file experiment_config/omniglot_maml++_omniglot_5_way_5_shot_maml_1.json --gpu_to_use 0