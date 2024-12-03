import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rich import print

from src.inner_loop_optimizers import LSLRGradientDescentLearningRule
from src.meta_architectures import VGGReLUNormNetwork


def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng


class MAMLFewShotClassifier(nn.Module):
    def __init__(self, im_shape, device, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLFewShotClassifier, self).__init__()
        self.args = args
        self.device = device
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.im_shape = im_shape
        self.current_epoch = 0

        self.rng = set_torch_seed(seed=args.seed)
        self.classifier = VGGReLUNormNetwork(
            im_shape=self.im_shape,
            num_output_classes=self.args.num_classes_per_set,
            args=args,
            device=device,
            meta_classifier=True,
        ).to(device=self.device)
        self.task_learning_rate = args.task_learning_rate

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(
            device=device,
            init_learning_rate=self.task_learning_rate,
            total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
            use_learnable_learning_rates=self.args.learnable_per_layer_per_step_inner_loop_learning_rate,
        )
        self.inner_loop_optimizer.initialise(
            names_weights_dict=self.get_inner_loop_parameter_dict(
                params=self.classifier.named_parameters()
            )
        )

        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)

        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self.to(device)
        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)

        self.optimizer = optim.Adam(
            self.trainable_parameters(),
            lr=args.meta_learning_rate,
            amsgrad=False,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.args.total_epochs,
            eta_min=self.args.min_learning_rate,
        )

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.to(torch.cuda.current_device())
                self.classifier = nn.DataParallel(module=self.classifier)
            else:
                self.to(torch.cuda.current_device())

            self.device = torch.cuda.current_device()

    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(
            shape=(self.args.number_of_training_steps_per_iter)
        ) * (1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = (
            1.0
            / self.args.number_of_training_steps_per_iter
            / self.args.multi_step_loss_num_epochs
        )
        min_value_for_non_final_losses = (
            0.03 / self.args.number_of_training_steps_per_iter
        )
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(
                loss_weights[i] - (self.current_epoch * decay_rate),
                min_value_for_non_final_losses,
            )
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1]
            + (
                self.current_epoch
                * (self.args.number_of_training_steps_per_iter - 1)
                * decay_rate
            ),
            1.0
            - (
                (self.args.number_of_training_steps_per_iter - 1)
                * min_value_for_non_final_losses
            ),
        )
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        return {
            name: param.to(device=self.device)
            for name, param in params
            if param.requires_grad
            and (
                not self.args.enable_inner_loop_optimizable_bn_params
                and "norm_layer" not in name
                or self.args.enable_inner_loop_optimizable_bn_params
            )
        }

    def apply_inner_loop_update(
        self, loss, names_weights_copy, use_second_order, current_step_idx
    ):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.classifier.module.zero_grad(params=names_weights_copy)
        else:
            self.classifier.zero_grad(params=names_weights_copy)

        grads = torch.autograd.grad(
            loss,
            names_weights_copy.values(),
            create_graph=use_second_order,
            allow_unused=True,
        )
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        names_weights_copy = {
            key: value[0] for key, value in names_weights_copy.items()
        }

        for key, grad in names_grads_copy.items():
            if grad is None:
                print("Grads not found for inner loop parameter", key)
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

        names_weights_copy = self.inner_loop_optimizer.update_params(
            names_weights_dict=names_weights_copy,
            names_grads_wrt_params_dict=names_grads_copy,
            num_step=current_step_idx,
        )

        num_devices = (
            torch.cuda.device_count() if torch.cuda.is_available() else 1
        )
        names_weights_copy = {
            name.replace("module.", ""): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]
            )
            for name, value in names_weights_copy.items()
        }

        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = {"loss": torch.mean(torch.stack(total_losses))}

        losses["accuracy"] = np.mean(total_accuracies)

        return losses

    def forward(
        self,
        data_batch,
        epoch,
        use_second_order,
        use_multi_step_loss_optimization,
        num_steps,
        training_phase,
    ):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        [b, ncs, spc] = y_support_set.shape

        self.num_classes_per_set = ncs

        total_losses = []
        total_accuracies = []
        per_task_target_preds = [[] for i in range(len(x_target_set))]
        self.classifier.zero_grad()
        task_accuracies = []
        for task_id, (
            x_support_set_task,
            y_support_set_task,
            x_target_set_task,
            y_target_set_task,
        ) in enumerate(
            zip(x_support_set, y_support_set, x_target_set, y_target_set)
        ):
            task_losses = []
            per_step_loss_importance_vectors = (
                self.get_per_step_loss_importance_vector()
            )
            names_weights_copy = self.get_inner_loop_parameter_dict(
                self.classifier.named_parameters()
            )

            num_devices = (
                torch.cuda.device_count() if torch.cuda.is_available() else 1
            )

            names_weights_copy = {
                name.replace("module.", ""): value.unsqueeze(0).repeat(
                    [num_devices] + [1 for i in range(len(value.shape))]
                )
                for name, value in names_weights_copy.items()
            }

            n, s, c, h, w = x_target_set_task.shape

            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            for num_step in range(num_steps):
                support_loss, support_preds = self.net_forward(
                    x=x_support_set_task,
                    y=y_support_set_task,
                    weights=names_weights_copy,
                    backup_running_statistics=num_step == 0,
                    training=training_phase,
                    num_step=num_step,
                )

                if training_phase:
                    # update inner loop parameters, iif we are training
                    names_weights_copy = self.apply_inner_loop_update(
                        loss=support_loss,
                        names_weights_copy=names_weights_copy,
                        use_second_order=use_second_order,
                        current_step_idx=num_step,
                    )

                if (
                    use_multi_step_loss_optimization
                    and training_phase
                    and epoch < self.args.multi_step_loss_num_epochs
                ):
                    target_loss, target_preds = self.net_forward(
                        x=x_target_set_task,
                        y=y_target_set_task,
                        weights=names_weights_copy,
                        backup_running_statistics=False,
                        training=training_phase,
                        num_step=num_step,
                    )

                    task_losses.append(
                        per_step_loss_importance_vectors[num_step]
                        * target_loss
                    )
                elif num_step == (
                    self.args.number_of_training_steps_per_iter - 1
                ):
                    target_loss, target_preds = self.net_forward(
                        x=x_target_set_task,
                        y=y_target_set_task,
                        weights=names_weights_copy,
                        backup_running_statistics=False,
                        training=training_phase,
                        num_step=num_step,
                    )
                    task_losses.append(target_loss)

            per_task_target_preds[task_id] = (
                target_preds.detach().cpu().numpy()
            )
            _, predicted = torch.max(target_preds.data, 1)

            accuracy = (
                predicted.float()
                .eq(y_target_set_task.data.float())
                .cpu()
                .float()
            )
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.extend(accuracy)

            if not training_phase:
                self.classifier.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(
            total_losses=total_losses, total_accuracies=total_accuracies
        )

        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses["loss_importance_vector_{}".format(idx)] = (
                item.detach().cpu().numpy()
            )

        return losses, per_task_target_preds

    def net_forward(
        self, x, y, weights, backup_running_statistics, training, num_step
    ):
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape b, c, h, w
        :param y: A data targets batch of shape b, n_classes
        :param weights: A dictionary containing the weights to pass to the network.
        :param backup_running_statistics: A flag indicating whether to reset the batch norm running statistics to their
         previous values after the run (only for evaluation)
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """
        preds = self.classifier.forward(
            x=x,
            params=weights,
            training=training,
            backup_running_statistics=backup_running_statistics,
            num_step=num_step,
        )

        loss = F.cross_entropy(input=preds, target=y)

        return loss, preds

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(
            data_batch=data_batch,
            epoch=epoch,
            use_second_order=self.args.second_order
            and epoch > self.args.first_order_to_second_order_epoch,
            use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
            num_steps=self.args.number_of_training_steps_per_iter,
            training_phase=True,
        )
        return losses, per_task_target_preds

    def evaluation_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(
            data_batch=data_batch,
            epoch=epoch,
            use_second_order=False,
            use_multi_step_loss_optimization=True,
            num_steps=self.args.number_of_evaluation_steps_per_iter,
            training_phase=False,
        )

        return losses, per_task_target_preds

    def meta_update(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        if "imagenet" in self.args.dataset_name:
            for name, param in self.classifier.named_parameters():
                if param.requires_grad:
                    param.grad.data.clamp_(
                        -10, 10
                    )  # not sure if this is necessary, more experiments are needed
        self.optimizer.step()

    def run_train_iter(self, data_batch, epoch):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        epoch = int(epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = (
            torch.Tensor(x_support_set).float().to(device=self.device)
        )
        x_target_set = (
            torch.Tensor(x_target_set).float().to(device=self.device)
        )
        y_support_set = (
            torch.Tensor(y_support_set).long().to(device=self.device)
        )
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_target_preds = self.train_forward_prop(
            data_batch=data_batch, epoch=epoch
        )

        self.meta_update(loss=losses["loss"])
        self.scheduler.step(epoch=epoch)
        losses["learning_rate"] = self.scheduler.get_last_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, per_task_target_preds

    def run_validation_iter(self, data_batch):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        if self.training:
            self.eval()

        with torch.no_grad():
            x_support_set, x_target_set, y_support_set, y_target_set = data_batch

            x_support_set = (
                torch.Tensor(x_support_set).float().to(device=self.device)
            )
            x_target_set = (
                torch.Tensor(x_target_set).float().to(device=self.device)
            )
            y_support_set = (
                torch.Tensor(y_support_set).long().to(device=self.device)
            )
            y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

            data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

            losses, per_task_target_preds = self.evaluation_forward_prop(
                data_batch=data_batch, epoch=self.current_epoch
            )

        # losses['loss'].backward() # uncomment if you get the weird memory error
        # self.zero_grad()
        # self.optimizer.zero_grad()

        return losses, per_task_target_preds

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        state["network"] = self.state_dict()
        state["optimizer"] = self.optimizer.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        """
        Load checkpoint and return the state dictionary containing the network state params and experiment state.
        :param model_save_dir: The directory from which to load the files.
        :param model_name: The model_name to be loaded from the direcotry.
        :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
        experiment)
        :return: A dictionary containing the experiment state and the saved model parameters.
        """
        filepath = os.path.join(
            model_save_dir, "{}_{}".format(model_name, model_idx)
        )
        state = torch.load(filepath)
        state_dict_loaded = state["network"]
        self.optimizer.load_state_dict(state["optimizer"])
        self.load_state_dict(state_dict=state_dict_loaded)
        return state
