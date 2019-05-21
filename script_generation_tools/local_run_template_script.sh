#!/bin/sh
cd ..
export DATASET_DIR="datasets/"
# Activate the relevant virtual environment:

python $execution_script$ --name_of_args_json_file experiment_config/$experiment_config$ --gpu_to_use 0