import os
from copy import copy

local_script_dir = "../experiment_scripts"
experiment_json_dir = '../experiment_config/'
maml_experiment_script = 'train_maml_system.py'
prefix = 'few_shot'

if not os.path.exists(local_script_dir):
    os.makedirs(local_script_dir)

def load_template(filepath):
    with open(filepath, mode='r') as filereader:
        template = filereader.readlines()

    return template

def fill_template(template_list, execution_script, experiment_config):
    template_list = copy(template_list)
    execution_line = template_list[-1]
    execution_line = execution_line.replace('$execution_script$', execution_script)
    execution_line = execution_line.replace('$experiment_config$', experiment_config)
    template_list[-1] = execution_line
    script_text = ''.join(template_list)

    return script_text

def write_text_to_file(text, filepath):
    with open(filepath, mode='w') as filewrite:
        filewrite.write(text)

local_script_template = load_template('local_run_template_script.sh')

for subdir, dir, files in os.walk(experiment_json_dir):
    for file in files:
        if file.endswith('.json'):
            config = file

            experiment_script = maml_experiment_script

            local_script_text = fill_template(template_list=local_script_template,
                                                execution_script=experiment_script,
                                                experiment_config=file)

            local_script_name = '../{}/{}_{}.sh'.format('experiment_scripts', file.replace(".json", ''), prefix)
            local_script_name = os.path.abspath(local_script_name)
            write_text_to_file(text=local_script_text, filepath=local_script_name)
