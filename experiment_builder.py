import tqdm
import os
import numpy as np
import sys
from utils.storage import build_experiment_folder, save_statistics, save_to_json
import time
import torch


class ExperimentBuilder(object):
    def __init__(self, args, data, model, device):
        """
        Initializes an experiment builder using a named tuple (args), a data provider (data), a meta learning system
        (model) and a device (e.g. gpu/cpu/n)
        :param args: A namedtuple containing all experiment hyperparameters
        :param data: A data provider of instance MetaLearningSystemDataLoader
        :param model: A meta learning system instance
        :param device: Device/s to use for the experiment
        """
        self.args, self.device = args, device

        self.model = model
        self.saved_models_filepath, self.logs_filepath, self.samples_filepath = build_experiment_folder(
            experiment_name=self.args.experiment_name)

        self.total_losses = dict()
        self.state = dict()
        self.state['best_val_acc'] = 0.
        self.state['best_val_iter'] = 0
        self.state['current_iter'] = 0
        self.state['current_iter'] = 0

        # Train from scratch or continue from checkpoint
        self.start_epoch = 0
        self.create_summary_csv = False

        if self.args.continue_from_epoch == 'from_scratch':
            print("Train from scratch.")
            self.create_summary_csv = True

        elif self.args.continue_from_epoch == 'latest' or \
                (self.args.continue_from_epoch.isdigit() and int(self.args.continue_from_epoch) >= 1):
            print(f"Attempting to find checkpoint for epoch {self.args.continue_from_epoch}... ", end='')
            if os.path.exists(os.path.join(self.saved_models_filepath, "train_model_latest")):
                print("succeeded")
                self.state = \
                    self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                          model_idx=self.args.continue_from_epoch)
                self.start_epoch = int(self.state['current_iter'] / self.args.total_iter_per_epoch)
            else:
                print("failed. Train from scratch.")
                self.args.continue_from_epoch = 'from_scratch'
                self.create_summary_csv = True

        else:
            raise ValueError(f"continue_from_epoch argument value invalid: {self.args.continue_from_epoch}")

        # Initialize dataloader
        self.data = data(args=args, current_iter=self.state['current_iter'])

        # Other initializations
        self.total_epochs_before_pause = self.args.total_epochs_before_pause
        self.state['best_epoch'] = int(self.state['best_val_iter'] / self.args.total_iter_per_epoch)
        self.epoch = self.state['current_iter'] // self.args.total_iter_per_epoch
        self.augment_flag = 'omniglot' in self.args.dataset_name.lower()
        self.start_time = time.time()
        self.epochs_done_in_this_run = 0

        # Print status
        print("STARTING WITH SEED:")
        print(f'train seed: {self.data.dataset.seed["train"]}, val seed: {self.data.dataset.seed["val"]}\n')
        print("CURRENT PROGRESS:")
        print(f'{self.state["current_iter"]} of {int(self.args.total_iter_per_epoch * self.args.total_epochs)} iters\n')

    def build_summary_dict(self, total_losses, phase, summary_losses=None):
        """
        Builds/Updates a summary dict directly from the metric dict of the current iteration.
        :param total_losses: Current dict with total losses (not aggregations) from experiment
        :param phase: Current training phase
        :param summary_losses: Current summarised (aggregated/summarised) losses stats means, stdv etc.
        :return: A new summary dict with the updated summary statistics information.
        """
        if summary_losses is None:
            summary_losses = dict()

        for key in total_losses.keys():
            summary_losses[f"{phase}_{key}_mean"] = np.mean(total_losses[key])
            summary_losses[f"{phase}_{key}_std"] = np.std(total_losses[key])

        return summary_losses

    def build_loss_summary_string(self, summary_losses):
        """
        Builds a progress bar summary string given current summary losses dictionary
        :param summary_losses: Current summary statistics
        :return: A summary string ready to be shown to humans.
        """
        output_update = ""
        importance_vector = []
        for key, value in summary_losses.items():
            if "loss" in key or "accuracy" in key:
                output_update += f"{key}: {float(value):.4f} | "
            elif "importance_vector" in key:
                importance_vector.append(value.item())
            elif "learning_rate" in key:
                pass
            else:
                raise NotImplementedError(key)

        if self.args.use_multi_step_loss_optimization:
            output_update += "step_importance: [" + ', '.join(f"{i:.4f}" for i in importance_vector) + "] "

        return output_update

    def merge_two_dicts(self, first_dict, second_dict):
        """Given two dicts, merge them into a new dict as a shallow copy."""
        z = first_dict.copy()
        z.update(second_dict)
        return z

    def train_iteration(self, train_sample, total_losses, pbar_train):
        """
        Runs a training iteration, updates the progress bar and returns the total and current epoch train losses.
        :param train_sample: A sample from the data provider.
        :param total_losses: The current total losses dictionary to be updated.
        :param pbar_train: The progress bar of the training.
        :return: Updates and returns total_losses, train_losses
        """
        # Slice out [x_support_set, x_target_set, y_support_set, y_target_set]
        data_batch = train_sample[:4]

        losses, _ = self.model.run_train_iter(data_batch=data_batch, epoch=self.epoch)

        for key, value in losses.items():
            if key not in total_losses:
                total_losses[key] = [float(value)]
            else:
                total_losses[key].append(float(value))

        train_losses = self.build_summary_dict(total_losses=total_losses, phase="train")
        train_output_update = self.build_loss_summary_string(losses)

        pbar_train.update(1)
        pbar_train.set_description_str(f"Epoch {self.epoch+1:03d}/{self.total_epochs_before_pause} -> {train_output_update}")

        return train_losses, total_losses

    def evaluation_iteration(self, val_sample, total_losses, pbar_val, phase):
        """
        Runs a validation iteration, updates the progress bar and returns the total and current epoch val losses.
        :param val_sample: A sample from the data provider
        :param total_losses: The current total losses dictionary to be updated.
        :param pbar_val: The progress bar of the val stage.
        :return: The updated val_losses, total_losses
        """
        # Slice out [x_support_set, x_target_set, y_support_set, y_target_set]
        data_batch = val_sample[:4]

        losses, _ = self.model.run_validation_iter(data_batch=data_batch)
        for key, value in losses.items():
            if key not in total_losses:
                total_losses[key] = [float(value)]
            else:
                total_losses[key].append(float(value))

        val_losses = self.build_summary_dict(total_losses=total_losses, phase=phase)
        val_output_update = self.build_loss_summary_string(losses)

        pbar_val.update(1)
        pbar_val.set_description_str(f"Validation    -> {val_output_update}")

        return val_losses, total_losses

    def test_evaluation_iteration(self, test_sample, model_idx, per_model_per_batch_preds, pbar_test):
        """
        Runs a test iteration, updates the progress bar and returns the per model per batch predictions.
        :param test_sample: A sample from the data provider.
        :param model_idx: Index of the model.
        :param per_model_per_batch_preds: per_model_per_batch_preds[model_idx] = per task predictions
        :param pbar_test: The progress bar of the test stage.
        :return: The extended per_model_per_batch_preds
        """
        # Slice out [x_support_set, x_target_set, y_support_set, y_target_set]
        data_batch = test_sample[:4]

        losses, per_task_preds = self.model.run_validation_iter(data_batch=data_batch)

        per_model_per_batch_preds[model_idx].extend(list(per_task_preds))

        test_output_update = self.build_loss_summary_string(losses)

        pbar_test.update(1)
        pbar_test.set_description_str(f"Test -> {test_output_update}")

        return per_model_per_batch_preds

    def save_models(self, model, epoch, state):
        """
        Saves two separate instances of the current model. One to be kept for history and reloading later and another
        one marked as "latest" to be used by the system for the next epoch training. Useful when the training/val
        process is interrupted or stopped. Leads to fault tolerant training and validation systems that can continue
        from where they left off before.
        :param model: Current meta learning model of any instance within the few_shot_learning_system.py
        :param epoch: Current epoch
        :param state: Current model and experiment state dict.
        """
        model.save_model(model_save_dir=os.path.join(self.saved_models_filepath, "train_model_{}".format(int(epoch))),
                         state=state)

        model.save_model(model_save_dir=os.path.join(self.saved_models_filepath, "train_model_latest"),
                         state=state)

    def pack_and_save_metrics(self, start_time, train_losses, val_losses, state, pbar_train):
        """
        Given current epochs start_time, train losses, val losses and whether to create a new stats csv file, pack stats
        and save into a statistics csv file. Return a new start time for the new epoch.
        :param start_time: The start time of the current epoch
        :param train_losses: A dictionary with the current train losses
        :param val_losses: A dictionary with the currrent val loss
        :param state: The current state of the experiment
        :param pbar_train: Progress bar for training phase
        :return: The current time, to be used for the next epoch.
        """
        epoch_summary_losses = self.merge_two_dicts(first_dict=train_losses, second_dict=val_losses)

        if 'per_epoch_statistics' not in state:
            state['per_epoch_statistics'] = dict()

        for key, value in epoch_summary_losses.items():
            if key not in state['per_epoch_statistics']:
                state['per_epoch_statistics'][key] = [value]
            else:
                state['per_epoch_statistics'][key].append(value)

        epoch_summary_string = self.build_loss_summary_string(epoch_summary_losses)
        epoch_summary_losses["epoch"] = self.epoch
        epoch_summary_losses['epoch_run_time'] = time.time() - start_time

        # Write header row for csv
        if self.create_summary_csv:
            self.summary_statistics_filepath = save_statistics(self.logs_filepath, list(epoch_summary_losses.keys()),
                                                               create=True)
            self.create_summary_csv = False

        start_time = time.time()
        pbar_train.write("epoch {} -> {}".format(epoch_summary_losses["epoch"], epoch_summary_string))

        self.summary_statistics_filepath = save_statistics(self.logs_filepath,
                                                           list(epoch_summary_losses.values()))
        return start_time, state

    def evaluate_test_set_using_the_best_models(self, top_n):
        # Choose the best top_n models
        per_epoch_statistics = self.state['per_epoch_statistics']
        val_acc = np.copy(per_epoch_statistics['val_accuracy_mean'])
        val_idx = np.arange(len(val_acc))
        top_n_idx = np.argsort(val_acc, axis=0).astype(dtype=np.int32)[::-1][:top_n]
        sorted_val_acc = val_acc[top_n_idx]

        print(f"\nTop {top_n} models chosen:")
        print(f"Epoch numbers {[idx + 1 for idx in top_n_idx]}")
        print(f"Mean validation accuracies {sorted_val_acc}")

        per_model_per_batch_preds = [[] for _ in range(top_n)]
        per_model_per_batch_targets = [[] for _ in range(top_n)]
        test_losses = [dict() for _ in range(top_n)]

        # Forward test set for each chosen model
        for idx, model_idx in enumerate(top_n_idx):
            self.state = \
                self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                      model_idx=model_idx + 1)
            with tqdm.tqdm(total=int(self.args.num_evaluation_tasks / self.args.batch_size)) as pbar_test:
                for sample_idx, test_sample in enumerate(
                        self.data.get_test_batches(total_batches=int(self.args.num_evaluation_tasks / self.args.batch_size),
                                                   augment_images=False)):
                    per_model_per_batch_targets[idx].extend(np.array(test_sample[3]))
                    per_model_per_batch_preds = self.test_evaluation_iteration(test_sample=test_sample,
                                                                               model_idx=idx,
                                                                               per_model_per_batch_preds=per_model_per_batch_preds,
                                                                               pbar_test=pbar_test)
            print()

        per_batch_preds = np.mean(per_model_per_batch_preds, axis=0)
        per_batch_max = np.argmax(per_batch_preds, axis=2)
        per_batch_targets = np.array(per_model_per_batch_targets[0]).reshape(per_batch_max.shape)

        accuracy = np.mean(np.equal(per_batch_targets, per_batch_max))
        accuracy_std = np.std(np.equal(per_batch_targets, per_batch_max))

        test_losses = {"test_accuracy_mean": accuracy, "test_accuracy_std": accuracy_std}

        save_statistics(self.logs_filepath, list(test_losses.keys()), create=True, filename="test_summary.csv")

        summary_statistics_filepath = save_statistics(self.logs_filepath, list(test_losses.values()),
                                                      create=False, filename="test_summary.csv")

        print()
        print(test_losses)
        print("Saved test performance at", summary_statistics_filepath)

    def run_experiment(self):
        """
        Runs a full training experiment with evaluations of the model on the val set at every epoch. Furthermore,
        will return the test set evaluation results on the best performing validation model.
        """
        with tqdm.tqdm(initial=self.state['current_iter'],
                       total=int(self.args.total_iter_per_epoch * self.args.total_epochs)) as pbar_train:

            while self.state['current_iter'] < (self.args.total_epochs * self.args.total_iter_per_epoch) and \
                    not self.args.evaluate_on_test_set_only:

                # Train Loop
                for train_sample_idx, train_sample in enumerate(
                        self.data.get_train_batches(total_batches=int(self.args.total_iter_per_epoch *
                                                                      self.args.total_epochs) - self.state[
                                                                      'current_iter'],
                                                    augment_images=self.augment_flag)):
                    
                    # Includes forward and backward propagations.
                    train_losses, total_losses = self.train_iteration(
                        train_sample=train_sample,
                        total_losses=self.total_losses,
                        pbar_train=pbar_train)

                    self.state['current_iter'] += 1

                    # Perform validation once after each epoch
                    if self.state['current_iter'] % self.args.total_iter_per_epoch == 0:
                        total_losses = dict()
                        val_losses = dict()

                        pbar_val = tqdm.tqdm(total=int(self.args.num_evaluation_tasks / self.args.batch_size))
                        for val_sample in self.data.get_val_batches(
                                                        total_batches=int(self.args.num_evaluation_tasks / self.args.batch_size),
                                                        augment_images=False):
                            val_losses, total_losses = self.evaluation_iteration(val_sample=val_sample,
                                                                                 total_losses=total_losses,
                                                                                 pbar_val=pbar_val,
                                                                                 phase='val')
                        pbar_val.close()

                        if val_losses["val_accuracy_mean"] > self.state['best_val_acc']:
                            pbar_train.write(f"Best validation accuracy: {val_losses['val_accuracy_mean']}")
                            self.state['best_val_acc'] = val_losses["val_accuracy_mean"]
                            self.state['best_val_iter'] = self.state['current_iter']
                            self.state['best_epoch'] = int(
                                self.state['best_val_iter'] / self.args.total_iter_per_epoch)

                        self.epoch += 1
                        self.epochs_done_in_this_run += 1

                        temp_state = self.merge_two_dicts(first_dict=self.state, second_dict=train_losses)
                        self.state = self.merge_two_dicts(first_dict=temp_state, second_dict=val_losses)

                        # Save model and metrics
                        self.save_models(model=self.model, epoch=self.epoch, state=self.state)
                        self.start_time, self.state = self.pack_and_save_metrics(start_time=self.start_time,
                                                                                 train_losses=train_losses,
                                                                                 val_losses=val_losses,
                                                                                 state=self.state,
                                                                                 pbar_train=pbar_train)
                        save_to_json(filename=os.path.join(self.logs_filepath, "summary_statistics.json"),
                                     dict_to_store=self.state['per_epoch_statistics'])
                        pbar_train.write(f"Saved model to {self.saved_models_filepath}")

                        # Reset loss dictionary
                        self.total_losses = dict()

                        if self.epochs_done_in_this_run >= self.total_epochs_before_pause:
                            print("RNG SEED:")
                            print(f'train seed: {self.data.dataset.seed["train"]}, val seed: {self.data.dataset.seed["val"]}\n')
                            print("PAUSED TRAINING.")
                            sys.exit()

            self.evaluate_test_set_using_the_best_models(top_n=5)
