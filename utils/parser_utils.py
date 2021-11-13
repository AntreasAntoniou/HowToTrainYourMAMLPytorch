from torch import cuda


def get_args():
    import argparse
    import os
    import torch
    import json
    parser = argparse.ArgumentParser(description='Welcome to the MAML++ training and inference system')

    parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='Batch_size for experiment')
    parser.add_argument('--image_height', nargs="?", type=int, default=28)
    parser.add_argument('--image_width', nargs="?", type=int, default=28)
    parser.add_argument('--image_channels', nargs="?", type=int, default=1)
    parser.add_argument('--reset_stored_filepaths', type=str, default="False")
    parser.add_argument('--reverse_channels', type=str, default="False")
    parser.add_argument('--num_of_gpus', type=int, default=1)
    parser.add_argument('--indexes_of_folders_indicating_class', nargs='+', default=[-2, -3])
    parser.add_argument('--train_val_test_split', nargs='+', default=[0.73982737361, 0.26, 0.13008631319])
    parser.add_argument('--samples_per_iter', nargs="?", type=int, default=1)
    parser.add_argument('--labels_as_int', type=str, default="False")
    parser.add_argument('--seed', type=int, default=104)

    parser.add_argument('--gpu_to_use', type=int)
    parser.add_argument('--num_dataprovider_workers', nargs="?", type=int, default=4)
    parser.add_argument('--max_models_to_save', nargs="?", type=int, default=5)
    parser.add_argument('--dataset_name', type=str, default="omniglot_dataset")
    parser.add_argument('--dataset_path', type=str, default="datasets/omniglot_dataset")
    parser.add_argument('--reset_stored_paths', type=str, default="False")
    parser.add_argument('--experiment_name', nargs="?", type=str, )
    parser.add_argument('--architecture_name', nargs="?", type=str)
    parser.add_argument('--continue_from_epoch', nargs="?", type=str, default='latest', help='Continue from checkpoint of epoch')
    parser.add_argument('--dropout_rate_value', type=float, default=0.3, help='Dropout_rate_value')
    parser.add_argument('--num_target_samples', type=int, default=15, help='Dropout_rate_value')
    parser.add_argument('--second_order', type=str, default="False", help='Dropout_rate_value')
    parser.add_argument('--total_epochs', type=int, default=200, help='Number of epochs per experiment')
    parser.add_argument('--total_iter_per_epoch', type=int, default=500, help='Number of iters per epoch')
    parser.add_argument('--min_learning_rate', type=float, default=0.00001, help='Min learning rate')
    parser.add_argument('--meta_learning_rate', type=float, default=0.001, help='Learning rate of overall MAML system')
    parser.add_argument('--meta_opt_bn', type=str, default="False")
    parser.add_argument('--task_learning_rate', type=float, default=0.1, help='Learning rate per task gradient step')

    parser.add_argument('--norm_layer', type=str, default="batch_norm")
    parser.add_argument('--max_pooling', type=str, default="False")
    parser.add_argument('--per_step_bn_statistics', type=str, default="False")
    parser.add_argument('--num_classes_per_set', type=int, default=20, help='Number of classes to sample per set')
    parser.add_argument('--cnn_num_blocks', type=int, default=4, help='Number of classes to sample per set')
    parser.add_argument('--number_of_training_steps_per_iter', type=int, default=1, help='Number of classes to sample per set')
    parser.add_argument('--number_of_evaluation_steps_per_iter', type=int, default=1, help='Number of classes to sample per set')
    parser.add_argument('--cnn_num_filters', type=int, default=64, help='Number of classes to sample per set')
    parser.add_argument('--cnn_blocks_per_stage', type=int, default=1,
                        help='Number of classes to sample per set')
    parser.add_argument('--num_samples_per_class', type=int, default=1, help='Number of samples per set to sample')
    parser.add_argument('--name_of_args_json_file', type=str, default="None")

    args = parser.parse_args()
    args_dict = vars(args)
    if args.name_of_args_json_file is not "None":
        args_dict = extract_args_from_json(args.name_of_args_json_file, args_dict)

    for key in list(args_dict.keys()):

        if str(args_dict[key]).lower() == "true":
            args_dict[key] = True
        elif str(args_dict[key]).lower() == "false":
            args_dict[key] = False
        if key == "dataset_path":
            args_dict[key] = os.path.join(os.environ['DATASET_DIR'], args_dict[key])
            print(key, os.path.join(os.environ['DATASET_DIR'], args_dict[key]))

        print(key, args_dict[key], type(args_dict[key]))

    args = Bunch(args_dict)


    args.use_cuda = torch.cuda.is_available()
    if torch.cuda.is_available():  # checks whether a cuda gpu is available and whether the gpu flag is True
        device = torch.cuda.current_device()

        print("use GPU", device)
        print("GPU ID {}".format(torch.cuda.current_device()))

    else:
        print("use CPU")
        device = torch.device('cpu')  # sets the device to be CPU


    return args, device



class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

def extract_args_from_json(json_file_path, args_dict):
    import json
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        if "continue_from" not in key and "gpu_to_use" not in key:
            args_dict[key] = summary_dict[key]

    return args_dict





