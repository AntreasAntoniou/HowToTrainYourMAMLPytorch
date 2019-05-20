import os
from collections import namedtuple

from utils.parser_utils import Bunch

seed_list = [0, 1, 2]

hyper_config = namedtuple('hyperconfig', 'num_samples_per_class_range '
                                         'batch_size_range '
                                         'init_inner_loop_learning_rate_range num_filters num_classes_range')

config = namedtuple('config', 'dataset_name num_classes '
                              'samples_per_class '
                              'target_samples_per_class '
                              'batch_size '
                              'train_update_steps '
                              'val_update_steps '
                              'init_inner_loop_learning_rate '
                              'load_into_memory '
                              'learnable_bn_beta '
                              'learnable_bn_gamma '
                              'conv_padding '
                              'num_filters '
                              'experiment_name ')

configs_list = []

hyper_config_dict = {'omniglot': hyper_config(num_samples_per_class_range=[1, 5], num_classes_range=[20, 5],
                                              batch_size_range=[8], init_inner_loop_learning_rate_range=[0.1],
                                              num_filters=[64]),
                     'mini-imagenet': hyper_config(num_samples_per_class_range=[1, 5],
                                                   batch_size_range=[2], init_inner_loop_learning_rate_range=[0.01],
                                                   num_classes_range=[5], num_filters=[48])
                     }


def generate_combinations(config):
    combos = []

    for key, item in config._asdict().items():
        if len(combos) == 0:
            combos = [[i] for i in item]
        else:
            combos = [combo + [choice] for combo in combos for choice in item]

    named_configs = []
    key_list = config._asdict().keys()
    key_list = list(key_list)

    for combo in combos:
        temp_dict = dict()
        for j in range(len(combo)):
            temp_dict[key_list[j].replace('_range', '')] = combo[j]
        named_configs.append(temp_dict)
    return named_configs


for seed in seed_list:
    for experiment_dataset_name, hyper_config in hyper_config_dict.items():
        named_configs = generate_combinations(hyper_config)
        # print(experiment_dataset_name, len(named_configs))
        for named_config in named_configs:
            experiment_name = '{}_{}'.format(experiment_dataset_name,
                                             '_'.join([str(item) for item in list(named_config.values())]))
            # print(experiment_name)
            configs_list.append(config(experiment_name=experiment_name, dataset_name=experiment_dataset_name,
                                       num_classes=named_config['num_classes'],
                                       samples_per_class=named_config['num_samples_per_class'],
                                       target_samples_per_class=15 if experiment_dataset_name == 'mini-imagenet' else 1,
                                       batch_size=named_config['batch_size'], train_update_steps=5, val_update_steps=5,
                                       init_inner_loop_learning_rate=named_config['init_inner_loop_learning_rate'],
                                       load_into_memory=True,
                                       learnable_bn_gamma=True,
                                       learnable_bn_beta=True, num_filters=named_config['num_filters'],
                                       conv_padding=True
                                       ))
# print(len(configs_list))
experiment_templates_json_dir = '../experiment_template_config/'
experiment_config_target_json_dir = '../experiment_config/'

if not os.path.exists(experiment_config_target_json_dir):
    os.makedirs(experiment_config_target_json_dir)


def fill_template(script_text, config):
    for key, item in vars(config).items():
        script_text = script_text.replace("${}$".format(str(key)), str(item).lower())

    return script_text


def load_template(filepath):
    with open(filepath, mode='r') as filereader:
        template = filereader.read()

    return template


def write_text_to_file(text, filepath):
    with open(filepath, mode='w') as filewrite:
        filewrite.write(text)


for subdir, dir, files in os.walk(experiment_templates_json_dir):
    for template_file in files:
        for seed_idx in seed_list:
            filepath = os.path.join(subdir, template_file)

            if "omniglot" in filepath:
                search_name = "omniglot"
            elif "imagenet" in filepath:
                search_name = "mini-imagenet"

            for config_item in configs_list:
                config_item = config_item._asdict()
                config_item['train_seed'] = seed_idx
                config_item['val_seed'] = 0
                config_item['experiment_name'] = "{}_{}".format(config_item['experiment_name'],
                                                                config_item['train_seed'])
                config_item = Bunch(config_item)

                if config_item.dataset_name == search_name:
                    loaded_template_file = load_template(filepath=filepath)

                    cluster_script_text = fill_template(script_text=loaded_template_file,
                                                        config=config_item)

                    cluster_script_name = '{}/{}-{}.json'.format(experiment_config_target_json_dir,
                                                                 template_file.replace('.json', ''),
                                                                 config_item.experiment_name)
                    cluster_script_name = os.path.abspath(cluster_script_name)
                    write_text_to_file(cluster_script_text, filepath=cluster_script_name)
