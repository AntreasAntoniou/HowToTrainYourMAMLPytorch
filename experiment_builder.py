import tqdm
import os
import numpy as np
import sys
from utils.storage import build_experiment_folder, save_statistics
import time


class ExperimentBuilder(object):
    def __init__(self, args, data, model, device):
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
        self.start_epoch = 0
        self.max_models_to_save = self.args.max_models_to_save
        self.create_summary_csv = False


        if self.args.continue_from_epoch == 'from_scratch':
            self.create_summary_csv = True

        elif self.args.continue_from_epoch == 'latest':
            checkpoint = os.path.join(self.saved_models_filepath, "train_model_latest")
            print("attempting to find existing checkpoint", )
            if os.path.exists(checkpoint):
                self.state = \
                    self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                          model_idx='latest')
                self.start_epoch = int(self.state['current_iter'] / self.args.total_iter_per_epoch)

            else:
                self.args.continue_from_epoch = -1
                self.create_summary_csv = True
        elif int(self.args.continue_from_epoch) >= 0:
            self.state = \
                self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                      model_idx=self.args.continue_from_epoch)
            self.start_epoch = int(self.state['current_iter'] / self.args.total_iter_per_epoch)


        self.data = data(args=args, current_iter=self.state['current_iter'])

        print("train_seed {}, val_seed: {}, at start time".format(self.data.dataset.seed["train"],
                                                                  self.data.dataset.seed["val"]))
        self.total_epochs_before_pause = self.args.total_epochs_before_pause
        self.state['best_epoch'] = int(self.state['best_val_iter'] / self.args.total_iter_per_epoch)
        self.epoch = int(self.state['current_iter'] / self.args.total_iter_per_epoch)
        self.augment_flag = True if 'omniglot' in self.args.dataset_name else False
        self.start_time = time.time()
        self.epochs_done_in_this_run = 0
        print(self.state['current_iter'], int(self.args.total_iter_per_epoch * self.args.total_epochs))

    def build_summary_dict(self, total_losses, phase, summary_losses=None):
        if summary_losses is None:
            summary_losses = dict()

        for key in total_losses:
            summary_losses["{}_{}_mean".format(phase, key)] = np.mean(total_losses[key])
            summary_losses["{}_{}_std".format(phase, key)] = np.std(total_losses[key])

        return summary_losses

    def build_loss_summary_string(self, summary_losses):
        output_update = ""
        for key, value in zip(list(summary_losses.keys()), list(summary_losses.values())):
            if "loss" in key or "accuracy" in key:
                value = float(value)
                output_update += "{}: {:.4f}, ".format(key, value)

        return output_update

    def merge_two_dicts(self, first_dict, second_dict):
        """Given two dicts, merge them into a new dict as a shallow copy."""
        z = first_dict.copy()
        z.update(second_dict)
        return z

    def train_iteration(self, train_sample, sample_idx, epoch_idx, total_losses, current_iter, pbar_train):
        x_support_set, x_target_set, y_support_set, y_target_set = train_sample
        data_batch = (x_support_set[0, 0], x_target_set[0, 0], y_support_set[0, 0], y_target_set[0, 0])

        if sample_idx == 0:
            print("shape of data", data_batch[0].shape, data_batch[1].shape, data_batch[2].shape,
                  data_batch[3].shape)

        losses = self.model.run_train_iter(data_batch=data_batch, epoch=epoch_idx)

        for key, value in zip(list(losses.keys()), list(losses.values())):
            if key not in total_losses:
                total_losses[key] = [float(value)]
            else:
                total_losses[key].append(float(value))

        train_losses = self.build_summary_dict(total_losses=total_losses, phase="train")
        train_output_update = self.build_loss_summary_string(losses)

        pbar_train.update(1)
        pbar_train.set_description("training phase {} -> {}".format(self.epoch, train_output_update))

        current_iter += 1

        return total_losses, train_losses, current_iter

    def evaluation_iteration(self, val_sample, total_losses, pbar_val):
        x_support_set, x_target_set, y_support_set, y_target_set = val_sample
        data_batch = (
            x_support_set[0, 0], x_target_set[0, 0], y_support_set[0, 0], y_target_set[0, 0])

        losses = self.model.run_validation_iter(data_batch=data_batch)
        for key, value in zip(list(losses.keys()), list(losses.values())):
            if key not in total_losses:
                total_losses[key] = [float(value)]
            else:
                total_losses[key].append(float(value))

        val_losses = self.build_summary_dict(total_losses=total_losses, phase="val")
        val_output_update = self.build_loss_summary_string(losses)

        pbar_val.update(1)
        pbar_val.set_description("val_phase {} -> {}".format(self.epoch, val_output_update))

        return val_losses, total_losses

    def save_models(self, model, epoch, state):
        model.save_model(model_save_dir=os.path.join(self.saved_models_filepath, "train_model_{}".format(int(epoch))),
                         state=state)

        model.save_model(model_save_dir=os.path.join(self.saved_models_filepath, "train_model_latest"),
                         state=state)

        print("saved models to", self.saved_models_filepath)

    def pack_and_save_metrics(self, start_time, create_summary_csv, train_losses, val_losses):
        epoch_summary_losses = self.merge_two_dicts(first_dict=train_losses, second_dict=val_losses)
        epoch_summary_string = self.build_loss_summary_string(epoch_summary_losses)
        epoch_summary_losses["epoch"] = self.epoch
        epoch_summary_losses['epoch_run_time'] = time.time() - start_time

        if create_summary_csv:
            self.summary_statistics_filepath = save_statistics(self.logs_filepath, list(epoch_summary_losses.keys()),
                                                               create=True)
            self.create_summary_csv = False

        start_time = time.time()
        print("epoch {} -> {}".format(epoch_summary_losses["epoch"], epoch_summary_string))

        self.summary_statistics_filepath = save_statistics(self.logs_filepath,
                                                           list(epoch_summary_losses.values()))
        return start_time

    def run_experiment(self):

        with tqdm.tqdm(initial=self.state['current_iter'],
                       total=int(self.args.total_iter_per_epoch * self.args.total_epochs)) as pbar_train:

            while self.state['current_iter'] < (self.args.total_epochs * self.args.total_iter_per_epoch):
                better_val_model = False

                for train_sample_idx, train_sample in enumerate(
                        self.data.get_train_batches(total_batches=int(self.args.total_iter_per_epoch *
                                                                      self.args.total_epochs) - self.state[
                                                                      'current_iter'],
                                                    augment_images=self.augment_flag)):
                    # print(self.state['current_iter'], (self.args.total_epochs * self.args.total_iter_per_epoch))
                    total_losses, train_losses, self.state['current_iter'] = self.train_iteration(
                        train_sample=train_sample,
                        total_losses=self.total_losses,
                        epoch_idx=(self.state['current_iter'] /
                                   self.args.total_iter_per_epoch),
                        pbar_train=pbar_train,
                        current_iter=self.state['current_iter'],
                        sample_idx=self.state['current_iter'])

                    if self.state['current_iter'] % self.args.total_iter_per_epoch == 0:

                        total_losses = dict()
                        val_losses = dict()
                        with tqdm.tqdm(total=self.args.total_iter_per_epoch) as pbar_val:
                            for _, val_sample in enumerate(
                                    self.data.get_val_batches(total_batches=int(self.args.total_iter_per_epoch),
                                                              augment_images=False)):
                                val_losses, total_losses = self.evaluation_iteration(val_sample=val_sample,
                                                                                     total_losses=total_losses,
                                                                                     pbar_val=pbar_val)

                            if val_losses["val_accuracy_mean"] > self.state['best_val_acc']:
                                print("Best validation accuracy", val_losses["val_accuracy_mean"])
                                self.state['best_val_acc'] = val_losses["val_accuracy_mean"]
                                self.state['best_val_iter'] = self.state['current_iter']
                                self.state['best_epoch'] = int(
                                    self.state['best_val_iter'] / self.args.total_iter_per_epoch)
                                better_val_model = True

                        self.epoch += 1
                        self.state = self.merge_two_dicts(first_dict=self.merge_two_dicts(first_dict=self.state,
                                                                                          second_dict=train_losses),
                                                          second_dict=val_losses)
                        self.save_models(model=self.model, epoch=self.epoch, state=self.state)

                        self.start_time = self.pack_and_save_metrics(start_time=self.start_time,
                                                                     create_summary_csv=self.create_summary_csv,
                                                                     train_losses=train_losses, val_losses=val_losses)

                        self.total_losses = dict()

                        self.epochs_done_in_this_run += 1

                        if self.epoch % 1 == 0 and better_val_model:
                            total_losses = dict()
                            test_losses = dict()
                            with tqdm.tqdm(total=self.args.total_iter_per_epoch) as pbar_test:
                                for _, test_sample in enumerate(
                                        self.data.get_test_batches(total_batches=int(self.args.total_iter_per_epoch),
                                                                   augment_images=False)):
                                    test_losses, total_losses = self.evaluation_iteration(val_sample=test_sample,
                                                                                          total_losses=total_losses,
                                                                                          pbar_val=pbar_test)

                            _ = save_statistics(self.logs_filepath,
                                                                          list(test_losses.keys()),
                                                                          create=True, filename="test_summary.csv")
                            summary_statistics_filepath = save_statistics(self.logs_filepath,
                                                                          list(test_losses.values()),
                                                                          create=False, filename="test_summary.csv")

                            print("saved test performance at", summary_statistics_filepath)

                        if self.epochs_done_in_this_run >= self.total_epochs_before_pause:
                            print("train_seed {}, val_seed: {}, at pause time".format(self.data.dataset.seed["train"],
                                                                                      self.data.dataset.seed["val"]))
                            sys.exit()
