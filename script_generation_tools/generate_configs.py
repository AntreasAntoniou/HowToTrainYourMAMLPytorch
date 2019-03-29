import os
from collections import namedtuple
seed_list = [0, 1, 2]

config = namedtuple('config', 'dataset_name num_classes '
                              'samples_per_class '
                              'target_samples_per_class '
                              'batch_size '
                              'train_update_steps '
                              'val_update_steps '
                              'inner_learning_rate')

configs_list = [config(dataset_name="omniglot", num_classes=20, samples_per_class=1, target_samples_per_class=1,
                       batch_size=16, train_update_steps=5, val_update_steps=5, inner_learning_rate=0.1),
                config(dataset_name="omniglot", num_classes=20, samples_per_class=1, target_samples_per_class=1,
                       batch_size=16, train_update_steps=1, val_update_steps=1, inner_learning_rate=0.1),
                config(dataset_name="omniglot", num_classes=20, samples_per_class=1, target_samples_per_class=1,
                       batch_size=16, train_update_steps=2, val_update_steps=2, inner_learning_rate=0.1),
                config(dataset_name="omniglot", num_classes=20, samples_per_class=1, target_samples_per_class=1,
                       batch_size=16, train_update_steps=3, val_update_steps=3, inner_learning_rate=0.1),
                config(dataset_name="omniglot", num_classes=20, samples_per_class=1, target_samples_per_class=1,
                       batch_size=16, train_update_steps=4, val_update_steps=4, inner_learning_rate=0.1),

                config(dataset_name="omniglot", num_classes=20, samples_per_class=5, target_samples_per_class=1,
                       batch_size=8, train_update_steps=5, val_update_steps=5, inner_learning_rate=0.1),

                config(dataset_name="omniglot", num_classes=5, samples_per_class=1, target_samples_per_class=1,
                       batch_size=16, train_update_steps=5, val_update_steps=5, inner_learning_rate=0.1),
                config(dataset_name="omniglot", num_classes=5, samples_per_class=5, target_samples_per_class=1,
                       batch_size=8, train_update_steps=5, val_update_steps=5, inner_learning_rate=0.1),

                config(dataset_name="mini_imagenet", num_classes=5, samples_per_class=1, target_samples_per_class=15,
                       batch_size=4, train_update_steps=5, val_update_steps=5, inner_learning_rate=0.01),
                config(dataset_name="mini_imagenet", num_classes=5, samples_per_class=1, target_samples_per_class=15,
                       batch_size=4, train_update_steps=1, val_update_steps=1, inner_learning_rate=0.01),
                config(dataset_name="mini_imagenet", num_classes=5, samples_per_class=1, target_samples_per_class=15,
                       batch_size=4, train_update_steps=2, val_update_steps=2, inner_learning_rate=0.01),
                config(dataset_name="mini_imagenet", num_classes=5, samples_per_class=1, target_samples_per_class=15,
                       batch_size=4, train_update_steps=3, val_update_steps=3, inner_learning_rate=0.01),
                config(dataset_name="mini_imagenet", num_classes=5, samples_per_class=1, target_samples_per_class=15,
                       batch_size=4, train_update_steps=4, val_update_steps=4, inner_learning_rate=0.01),

                config(dataset_name="mini_imagenet", num_classes=5, samples_per_class=5, target_samples_per_class=15,
                       batch_size=2, train_update_steps=5, val_update_steps=5, inner_learning_rate=0.01)]



experiment_templates_json_dir = '../experiment_template_config/'
experiment_config_target_json_dir = '../experiment_config/'

if not os.path.exists(experiment_config_target_json_dir):
    os.makedirs(experiment_config_target_json_dir)

def fill_template(script_text, train_seed, val_seed, batch_size, num_classes, samples_per_class,
                  target_samples_per_class, train_update_steps, val_update_steps, inner_learning_rate):
    script_text = script_text.replace('$train_seed$', str(train_seed))
    script_text = script_text.replace('$val_seed$', str(val_seed))
    script_text = script_text.replace('$batch_size$', str(batch_size))
    script_text = script_text.replace('$num_classes$', str(num_classes))
    script_text = script_text.replace('$samples_per_class$', str(samples_per_class))
    script_text = script_text.replace('$target_samples_per_class$', str(target_samples_per_class))
    script_text = script_text.replace('$train_update_steps$', str(train_update_steps))
    script_text = script_text.replace('$val_update_steps$', str(val_update_steps))
    script_text = script_text.replace('$inner_loop_learning_rate$', str(inner_learning_rate))

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
                search_name = "mini_imagenet"

            for config in configs_list:
                if config.dataset_name == search_name:
                    loaded_template_file = load_template(filepath=filepath)

                    cluster_script_text = fill_template(script_text=loaded_template_file,
                                                        train_seed=seed_idx,
                                                        val_seed=0, num_classes=config.num_classes,
                                                        batch_size=config.batch_size,
                                                        samples_per_class=config.samples_per_class,
                                                        target_samples_per_class=config.target_samples_per_class,
                                                        train_update_steps=config.train_update_steps,
                                                        val_update_steps=config.val_update_steps,
                                                        inner_learning_rate=config.inner_learning_rate)

                    cluster_script_name = '{}/{}_{}.json'.format(experiment_config_target_json_dir,
                                                                 template_file.replace(".json", '')
                                                                 .replace("samples_per_class",
                                                                          str(config.samples_per_class))
                                                                 .replace("num_classes",
                                                                          str(config.num_classes)).replace("num_steps",
                                                                          str(config.train_update_steps)), seed_idx)
                    cluster_script_name = os.path.abspath(cluster_script_name)
                    write_text_to_file(cluster_script_text, filepath=cluster_script_name)
