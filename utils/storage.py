import csv
import datetime
import os
import numpy as np
from utils.parser_utils import get_args
import json

def save_to_json(filename, dict_to_store):
    with open(os.path.abspath(filename), 'w') as f:
        json.dump(dict_to_store, fp=f)

def load_from_json(filename):
    with open(filename, mode="r") as f:
        load_dict = json.load(fp=f)

    return load_dict

def save_statistics(experiment_name, line_to_add, filename="summary_statistics.csv", create=False):
    summary_filename = "{}/{}".format(experiment_name, filename)
    if create:
        with open(summary_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(line_to_add)
    else:
        with open(summary_filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(line_to_add)

    return summary_filename

def load_statistics(experiment_name, filename="summary_statistics.csv"):
    data_dict = dict()
    summary_filename = "{}/{}".format(experiment_name, filename)
    with open(summary_filename, 'r') as f:
        lines = f.readlines()
        data_labels = lines[0].replace("\n", "").split(",")
        del lines[0]

        for label in data_labels:
            data_dict[label] = []

        for line in lines:
            data = line.replace("\n", "").split(",")
            for key, item in zip(data_labels, data):
                data_dict[key].append(item)
    return data_dict


def build_experiment_folder(experiment_name):
    experiment_path = os.path.abspath(experiment_name)
    saved_models_filepath = "{}/{}".format(experiment_path, "saved_models")
    logs_filepath = "{}/{}".format(experiment_path, "logs")
    samples_filepath = "{}/{}".format(experiment_path, "visual_outputs")

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    if not os.path.exists(logs_filepath):
        os.makedirs(logs_filepath)
    if not os.path.exists(samples_filepath):
        os.makedirs(samples_filepath)
    if not os.path.exists(saved_models_filepath):
        os.makedirs(saved_models_filepath)

    outputs = (saved_models_filepath, logs_filepath, samples_filepath)
    outputs = (os.path.abspath(item) for item in outputs)
    return outputs

def get_best_validation_model_statistics(experiment_name, filename="summary_statistics.csv"):
    """
    Returns the best val epoch and val accuracy from a log csv file
    :param log_dir: The log directory the file is saved in
    :param statistics_file_name: The log file name
    :return: The best validation accuracy and the epoch at which it is produced
    """
    log_file_dict = load_statistics(filename=filename, experiment_name=experiment_name)
    d_val_loss = np.array(log_file_dict['total_d_val_loss_mean'], dtype=np.float32)
    best_d_val_loss = np.min(d_val_loss)
    best_d_val_epoch = np.argmin(d_val_loss)

    return best_d_val_loss, best_d_val_epoch

def create_json_experiment_log(experiment_log_dir, args, log_name="experiment_log.json"):
    summary_filename = "{}/{}".format(experiment_log_dir, log_name)

    experiment_summary_dict = dict()

    for key, value in vars(args).items():
        experiment_summary_dict[key] = value

    experiment_summary_dict["epoch_stats"] = dict()
    timestamp = datetime.datetime.now().timestamp()
    experiment_summary_dict["experiment_status"] = [(timestamp, "initialization")]
    experiment_summary_dict["experiment_initialization_time"] = timestamp
    with open(os.path.abspath(summary_filename), 'w') as f:
        json.dump(experiment_summary_dict, fp=f)

def update_json_experiment_log_dict(key, value, experiment_log_dir, log_name="experiment_log.json"):
    summary_filename = "{}/{}".format(experiment_log_dir, log_name)
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)

    summary_dict[key].append(value)

    with open(summary_filename, 'w') as f:
        json.dump(summary_dict, fp=f)

def change_json_log_experiment_status(experiment_status, experiment_log_dir, log_name="experiment_log.json"):
    timestamp = datetime.datetime.now().timestamp()
    experiment_status = (timestamp, experiment_status)
    update_json_experiment_log_dict(key="experiment_status", value=experiment_status,
                                    experiment_log_dir=experiment_log_dir, log_name=log_name)

def update_json_experiment_log_epoch_stats(epoch_stats, experiment_log_dir, log_name="experiment_log.json"):
    summary_filename = "{}/{}".format(experiment_log_dir, log_name)
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)

    epoch_stats_dict = summary_dict["epoch_stats"]

    for key in epoch_stats.keys():
        entry = float(epoch_stats[key])
        if key in epoch_stats_dict:
            epoch_stats_dict[key].append(entry)
        else:
            epoch_stats_dict[key] = [entry]

    summary_dict['epoch_stats'] = epoch_stats_dict

    with open(summary_filename, 'w') as f:
        json.dump(summary_dict, fp=f)
    return summary_filename
