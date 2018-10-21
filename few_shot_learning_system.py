from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import time
import argparse
import datetime

from PIL import Image

from meta_neural_network_architectures import VGGLeakyReLUNormNetwork
from torch.autograd import Variable
import numpy as np


def set_torch_seed(seed):
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng


class MAMLFewShotClassifier(nn.Module):
    def __init__(self, im_shape, device, args):

        """

        """
        super(MAMLFewShotClassifier, self).__init__()
        self.args = args
        self.device = device
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.im_shape = im_shape
        self.current_epoch = 0

        self.rng = set_torch_seed(seed=args.seed)
        self.classifier = VGGLeakyReLUNormNetwork(im_shape=self.im_shape, num_output_classes=self.args.
                                                  num_classes_per_set,
                                                  args=args, device=device, meta_classifier=True).to(device=self.device)
        self.task_learning_rate = args.task_learning_rate

        if self.task_learning_rate == -1:
            names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())
            self.task_learning_rate = nn.Parameter(
                data=0.1 * torch.ones((self.args.number_of_training_steps_per_iter, len(names_weights_copy.keys()))),
                requires_grad=True)
        else:
            self.task_learning_rate = torch.ones((1)) * args.task_learning_rate

        self.task_learning_rate.to(device=self.device)

        task_name_params = self.get_inner_loop_parameter_dict(self.named_parameters())

        print("Inner Loop parameters")
        for key, value in task_name_params.items():
            print(key, value.shape)

        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args

        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape)

        self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)

    def get_per_step_loss_importance_vector(self):
        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter)) * (
                    1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.args.number_of_training_steps_per_iter / 15.
        min_value_for_non_final_losses = 0.03 / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-2] = curr_value
        loss_weights[-1] = np.sum(loss_weights[:-1])
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def get_inner_loop_parameter_dict(self, params):
        param_dict = dict()
        for name, param in params:
            if "norm_layer" in name:
                if param.requires_grad and self.args.meta_opt_bn:
                    param_dict[name] = param.to(device=self.device)

            else:
                if param.requires_grad:
                    param_dict[name] = param.to(device=self.device)

        return param_dict

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):

        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order)

        if len(self.task_learning_rate.shape) > 1:
            updated_weights = list(map(
                lambda p: p[1].to(device=self.device) - p[2].to(device=self.device) *
                          p[0].to(device=self.device),
                zip(grads, names_weights_copy.values(), self.task_learning_rate[current_step_idx])))
        else:
            updated_weights = list(map(
                lambda p: p[1].to(device=self.device) - self.task_learning_rate[0].to(device=self.device) * p[0].to(
                    device=self.device), zip(grads, names_weights_copy.values())))

        names_weights_copy = dict(zip(names_weights_copy.keys(), updated_weights))

        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies, names_weights_copy):
        losses = dict()

        losses['loss'] = torch.sum(torch.stack(total_losses))
        losses['accuracy'] = np.mean(total_accuracies)
        names_weights = list(names_weights_copy.keys())

        if len(self.task_learning_rate.shape) > 1:
            for idx_num_step, learning_rate_num_step in enumerate(self.task_learning_rate):
                for idx, learning_rate in enumerate(learning_rate_num_step):
                    losses['task_learning_rate_num_step_{}_{}'.format(idx_num_step,
                                                             names_weights[idx])] = learning_rate.detach().cpu().numpy()
        else:
            losses['task_learning_rate'] = self.task_learning_rate.detach().cpu().numpy()[0]

        return losses

    def forward(self, data_batch, epoch, use_second_order, optimize_final_target_loss_only, num_steps, training_phase):
        """

        :return:
        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        [b, ncs, spc] = y_support_set.shape

        self.num_classes_per_set = ncs

        total_losses = []
        total_accuracies = []

        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in \
                enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):
            task_losses = []
            task_accuracies = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
            names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

            n, s, c, h, w = x_target_set_task.shape

            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            for num_step in range(num_steps):
                support_loss, support_preds = self.net_forward(x=x_support_set_task,
                                                               y=y_support_set_task,
                                                               weights=names_weights_copy,
                                                               reset_running_statistics=
                                                               True if (num_step == 0) else False,
                                                               training=True, num_step=num_step)

                names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step)

                if optimize_final_target_loss_only:
                    if num_step == (self.args.number_of_training_steps_per_iter - 1):
                        target_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                                     y=y_target_set_task, weights=names_weights_copy,
                                                                     reset_running_statistics=False, training=True,
                                                                     num_step=num_step)
                        task_losses.append(target_loss)
                else:
                    target_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                                 y=y_target_set_task, weights=names_weights_copy,
                                                                 reset_running_statistics=False, training=True,
                                                                 num_step=num_step)

                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)


            _, predicted = torch.max(target_preds.data, 1)
            accuracy = list(predicted.eq(y_target_set_task.data).cpu())
            task_accuracies.extend(accuracy)
            task_accuracies = np.mean(task_accuracies)
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.append(task_accuracies)

            if not training_phase:
                self.classifier.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   total_accuracies=total_accuracies,
                                                   names_weights_copy=names_weights_copy)

        return losses

    def net_forward(self, x, y, weights, reset_running_statistics, training, num_step):
        ''' Run data through net, return loss and output '''
        input_var = torch.autograd.Variable(x).to(device=self.device)
        target_var = torch.autograd.Variable(y).to(device=self.device)
        # Run the batch through the net, compute loss
        preds = self.classifier.forward(x=input_var, params=weights,
                                        training=training,
                                        reset_running_statistics=reset_running_statistics, num_step=num_step)

        target_preds = preds
        loss = F.cross_entropy(input=target_preds, target=target_var)

        return loss, target_preds

    def trainable_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_iter(self, data_batch, epoch):
        # print("\n second_order", self.args.second_order and
        #       epoch > self.args.first_order_to_second_order_epoch)
        losses = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=self.args.second_order and
                                                                                   epoch > self.args.first_order_to_second_order_epoch,
                              optimize_final_target_loss_only=self.args.optimize_final_target_loss_only,
                              num_steps=self.args.number_of_training_steps_per_iter, training_phase=True)
        return losses

    def evaluation_iter(self, data_batch, epoch):
        """
        Builds tf graph for Matching Networks, produces losses and summary statistics.
        :return:
        """
        losses = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                              optimize_final_target_loss_only=True,
                              num_steps=self.args.number_of_evaluation_steps_per_iter, training_phase=False)

        return losses

    def meta_update(self, loss):

        self.optimizer.zero_grad()
        loss.backward()
        if 'imagenet' in self.args.dataset_name:
            for name, param in self.classifier.named_parameters():
                if param.requires_grad:
                    # print(name)
                    param.grad.data.clamp_(-10, 10)
        # Update the net parameters with the accumulated gradient according to optimizer
        self.optimizer.step()

    def run_train_iter(self, data_batch, epoch):
        """
        Builds the train op
        :param losses: A dictionary containing the losses
        :param learning_rate: Learning rate to be used for Adam
        :param beta1: Beta1 to be used for Adam
        :return:
        """

        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses = self.train_iter(data_batch=data_batch, epoch=epoch)

        self.meta_update(loss=losses['loss'])
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()

        return losses

    def run_validation_iter(self, data_batch):
        """
        Builds the train op
        :param losses: A dictionary containing the losses
        :param learning_rate: Learning rate to be used for Adam
        :param beta1: Beta1 to be used for Adam
        :return:
        """

        if self.training:
            self.eval()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses = self.evaluation_iter(data_batch=data_batch, epoch=self.current_epoch)

        self.zero_grad()
        self.optimizer.zero_grad()

        # for name, param in self.named_parameters():
        #     if param.grad is not None and torch.sum(param.grad) > 0:
        #         print(name, param.shape, param.grad)

        return losses

    def save_model(self, model_save_dir, state):
        state['network'] = self.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        checkpoint_state = torch.load(filepath)
        state_dict_loaded = checkpoint_state['network']
        self.load_state_dict(state_dict=state_dict_loaded)
        return checkpoint_state
