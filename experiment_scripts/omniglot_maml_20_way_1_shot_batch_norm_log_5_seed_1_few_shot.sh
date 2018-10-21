#!/bin/sh
cd ..
export DATASET_DIR="datasets/"
# Activate the relevant virtual environment:
#python check_download_dataset.py --name_of_args_json_file experiment_config/umaml_maml_omniglot_characters_20_1_seed_1.json
python train_maml_system.py --name_of_args_json_file experiment_config/omniglot_maml_20_way_1_shot_batch_norm_log_5_seed_1.json --gpu_to_use 0