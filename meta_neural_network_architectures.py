import copy
import numbers
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class BatchNormReLUConv(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, use_bias):
        super(BatchNormReLUConv, self).__init__()

        self.bn = nn.BatchNorm2d(input_shape[1], track_running_stats=True)
        self.conv = MetaConv2dLayer(in_channels=input_shape[1], out_channels=num_filters,
                                    kernel_size=kernel_size, stride=stride, padding=padding, use_bias=use_bias)

    def forward(self, x):
        out = self.conv(F.leaky_relu(self.bn(x)))
        return out


class MetaConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias):
        super(MetaConv2dLayer, self).__init__()
        num_filters = out_channels
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size, kernel_size))
        nn.init.xavier_normal_(self.weight)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        if params is not None:

            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
            else:
                (weight) = params["weight"]
                bias = None
        else:
            if self.use_bias:
                weight, bias = self.weight, self.bias
            else:
                weight = self.weight
                bias = None

        out = F.conv2d(input=x, weight=weight, bias=bias, stride=self.stride,
                       padding=self.padding, dilation=1, groups=1)
        return out

class MetaLinearLayer(nn.Module):
    def __init__(self, input_shape, num_filters, use_bias):
        super(MetaLinearLayer, self).__init__()
        b, c = input_shape
        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.empty(num_filters, c))
        nn.init.normal(self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        if params is not None:
            if self.use_bias:
                (weight, bias) = params["weights"], params["bias"]
            else:
                (weight) = params["weights"]
                bias = None
        else:
            if self.use_bias:
                weight, bias = self.weights, self.bias
            else:
                weight = self.weights
                bias = None
        out = F.linear(input=x, weight=weight, bias=bias)
        return out

class MetaBatchNormLayer(nn.Module):
    def __init__(self, num_features, device, args, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, meta_batch_norm=True, no_learnable_params=False,
                 use_per_step_bn_statistics=False):
        super(MetaBatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.affine = affine
        self.track_running_stats = track_running_stats
        self.meta_batch_norm = meta_batch_norm
        self.num_features = num_features
        self.device = device
        self.use_per_step_bn_statistics = use_per_step_bn_statistics
        self.args = args

        if no_learnable_params:
            self.weight = nn.Parameter(torch.ones(num_features), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=False)
        elif meta_batch_norm:
            self.weight = nn.Parameter(torch.ones(num_features), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        else:
            self.weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)

        if use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features),
                                             requires_grad=False)
            self.running_var = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features),
                                            requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features),
                                     requires_grad=True)
            self.weight = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features),
                                       requires_grad=True)
        else:
            self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.running_var = nn.Parameter(torch.ones(num_features), requires_grad=False)

        self.backup_running_mean = self.running_mean.clone()
        self.backup_running_var = self.running_var.clone()

        self.momentum = momentum

    def forward(self, input, num_step, params=None, training=False, reset_running_statistics=False):

        if params is not None:
            (weight, bias) = params["weight"], params["bias"]
        else:
            weight, bias = self.weight, self.bias

        if self.use_per_step_bn_statistics:
            running_mean = self.running_mean[num_step]
            running_var = self.running_var[num_step]
            bias = bias[num_step]
            weight = weight[num_step]
        else:
            running_mean = None
            running_var = None

        if reset_running_statistics and self.use_per_step_bn_statistics:
            self.backup_running_mean[num_step] = self.running_mean[num_step].clone()
            self.backup_running_var[num_step] = self.running_var[num_step].clone()

        momentum = self.momentum

        output = F.batch_norm(input, running_mean, running_var, weight, bias,
                              training=True, momentum=momentum, eps=self.eps)

        return output

    def restore_backup_stats(self):
        if self.use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(self.backup_running_mean.to(device=self.device), requires_grad=False)
            self.running_var = nn.Parameter(self.backup_running_var.to(device=self.device), requires_grad=False)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class MetaLayerNormLayer(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(MetaLayerNormLayer, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape), requires_grad=False)
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input, num_step, params=None, training=False, reset_running_statistics=False):

        if params is not None:
            bias = params["bias"]
        else:
            bias = self.bias

        return F.layer_norm(
            input, self.normalized_shape, self.weight, bias, self.eps)

    def restore_backup_stats(self):
        pass

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class MetaConvReLUNormLayer(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, use_bias, args, device, normalization=True,
                 meta_layer=True):
        super(MetaConvReLUNormLayer, self).__init__()
        self.meta_layer = meta_layer
        self.normalization = normalization
        self.conv = MetaConv2dLayer(in_channels=input_shape[1], out_channels=num_filters, kernel_size=kernel_size,
                                    stride=stride, padding=padding, use_bias=use_bias)
        if normalization:
            if args.norm_layer == "batch_norm":
                self.norm_layer = MetaBatchNormLayer(num_filters, track_running_stats=True,
                                                     meta_batch_norm=self.meta_layer, args=args, device=device)
            elif args.norm_layer == "layer_norm":
                input_shape_list = list(input_shape)
                input_shape_list[1] = num_filters
                b, c, h, w = input_shape
                input_shape_list[2] = int(np.ceil(input_shape_list[2] / stride))
                input_shape_list[3] = int(np.ceil(input_shape_list[3] / stride))
                self.norm_layer = MetaLayerNormLayer(normalized_shape=input_shape_list[1:])

        self.total_layers = 1

    def forward(self, x, params=None, training=False, reset_running_statistics=False):

        batch_norm_params = None

        if params is not None:

            if self.normalization:
                if 'norm_layer' in params:
                    batch_norm_params = params['norm_layer']

            conv_params = params['conv']
        else:
            conv_params = None

        if self.normalization:
            out = self.norm_layer.forward(F.leaky_relu(self.conv.forward(x, conv_params)),
                                          params=batch_norm_params, training=training,
                                          reset_running_statistics=reset_running_statistics)
        else:
            out = F.leaky_relu(self.conv(x, params=conv_params))
        return out


class MetaNormLayerConvReLU(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, use_bias, args, normalization=True,
                 meta_layer=True, no_bn_learnable_params=False, device=None):
        super(MetaNormLayerConvReLU, self).__init__()
        self.normalization = normalization
        self.use_per_step_bn_statistics = args.per_step_bn_statistics
        print('use_per_step_bn_stats', self.use_per_step_bn_statistics)
        if normalization:
            if args.norm_layer == "batch_norm":
                self.norm_layer = MetaBatchNormLayer(input_shape[1], track_running_stats=True,
                                                     meta_batch_norm=meta_layer,
                                                     no_learnable_params=no_bn_learnable_params, device=device,
                                                     use_per_step_bn_statistics=self.use_per_step_bn_statistics,
                                                     args=args)
            elif args.norm_layer == "layer_norm":
                input_shape_list = list(input_shape)
                self.norm_layer = MetaLayerNormLayer(normalized_shape=input_shape_list[1:])

        self.conv = MetaConv2dLayer(in_channels=input_shape[1], out_channels=num_filters, kernel_size=kernel_size,
                                    stride=stride, padding=padding, use_bias=use_bias)

        self.total_layers = 1

    def forward(self, x, num_step, params=None, training=False, reset_running_statistics=False):

        batch_norm_params = None

        if params is not None:

            if self.normalization:
                if 'norm_layer' in params:
                    batch_norm_params = params['norm_layer']

            conv_params = params['conv']
        else:
            conv_params = None

        if self.normalization:
            out = self.norm_layer.forward(x, num_step=num_step,
                                          params=batch_norm_params, training=training,
                                          reset_running_statistics=reset_running_statistics)

            out = self.conv.forward(out, conv_params)
            out = F.leaky_relu(out)
        else:
            out = F.leaky_relu(x)
            out = self.conv(out, params=conv_params)

        return out

    def restore_backup_stats(self):
        if self.normalization:
            self.norm_layer.restore_backup_stats()


class VGGLeakyReLUNormNetwork(nn.Module):
    def __init__(self, im_shape, num_output_classes, args, device, meta_classifier=True):
        super(VGGLeakyReLUNormNetwork, self).__init__()
        b, c, self.h, self.w = im_shape
        self.device = device
        self.total_layers = 0
        self.args = args
        self.upscale_shapes = []
        self.cnn_filters = args.cnn_num_filters
        self.input_shape = list(im_shape)
        self.num_stages = args.num_stages
        self.num_output_classes = num_output_classes

        if args.max_pooling:
            print("Using max pooling")
            self.conv_stride = 1
        else:
            print("Using strided convolutions")
            self.conv_stride = 2
        self.meta_classifier = meta_classifier

        self.build_network()
        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)

    def build_network(self):
        x = torch.zeros(self.input_shape)
        out = x
        self.layer_dict = nn.ModuleDict()
        self.upscale_shapes.append(x.shape)
        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)] = MetaNormLayerConvReLU(input_shape=out.shape,
                                                                        num_filters=self.cnn_filters,
                                                                        kernel_size=3, stride=self.conv_stride,
                                                                        padding=1,
                                                                        use_bias=True, args=self.args,
                                                                        normalization=False if i == 0 else True,
                                                                        meta_layer=self.meta_classifier,
                                                                        no_bn_learnable_params=False,
                                                                        device=self.device)
            out = self.layer_dict['conv{}'.format(i)](out, training=True, num_step=0)

            if self.args.max_pooling:
                out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)

        if self.args.norm_layer == "batch_norm":
            self.layer_dict['fc_norm_layer'] = MetaBatchNormLayer(out.shape[1], track_running_stats=True,
                                                                  meta_batch_norm=self.meta_classifier,
                                                                  device=self.device, args=self.args,
                                                                  use_per_step_bn_statistics=self.args.per_step_bn_statistics)
            out = self.layer_dict['fc_norm_layer'](out, training=True, num_step=0)

        elif self.args.norm_layer == "layer_norm":
            input_shape_list = list(out.shape)
            self.layer_dict['fc_norm_layer'] = MetaLayerNormLayer(normalized_shape=input_shape_list[1:])
            out = self.layer_dict['fc_norm_layer'](out, num_step=0)

        out = F.leaky_relu(out)

        if not self.args.max_pooling:
            out = F.avg_pool2d(out, out.shape[2])

        self.layer_dict['linear'] = MetaLinearLayer(input_shape=(out.shape[0], np.prod(out.shape[1:])),
                                                    num_filters=self.num_output_classes, use_bias=True)

        self.encoder_features_shape = list(out.shape)
        out = out.view(out.shape[0], -1)

        out = self.layer_dict['linear'](out)

        return out

    def create_new_nested_dictionary(self, depth_keys, value, key_exists=None):
        temp_current_dictionary = {depth_keys[-1]: value}

        if key_exists is not None:
            for idx, key in enumerate(depth_keys[:-1]):
                key_exists = key_exists[key]

            for key, item in key_exists.items():
                temp_current_dictionary[key] = item


        for idx, key in enumerate(depth_keys[::-1]):
            if idx>0:
                temp_current_dictionary[key] = temp_current_dictionary

        return temp_current_dictionary

    def forward(self, x, num_step, params=None, training=False, reset_running_statistics=False,
                output_only_conv_features=False, output_every_layer_features=False):
        param_dict = dict()

        if params is not None:
            for name, param in params.items():
                path_bits = name.split(".")
                if path_bits[1] not in param_dict:
                    param_dict[path_bits[1]] = self.create_new_nested_dictionary(depth_keys=path_bits[2:], value=param)
                else:
                    param_dict[path_bits[1]] = self.create_new_nested_dictionary(depth_keys=path_bits[2:], value=param,
                                                                                 key_exists=param_dict[path_bits[1]])

        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        for i in range(self.num_stages):
            out = self.layer_dict['conv{}'.format(i)](out, params=param_dict['conv{}'.format(i)], training=training,
                                                      reset_running_statistics=reset_running_statistics,
                                                      num_step=num_step)
            if self.args.max_pooling:
                out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)

        if self.args.norm_layer == "batch_norm":
            out = self.layer_dict['fc_norm_layer'](out, num_step=num_step, params=param_dict['fc_norm_layer'],
                                                   training=training,
                                                   reset_running_statistics=reset_running_statistics)

        elif self.args.norm_layer == "layer_norm":
            out = self.layer_dict['fc_norm_layer'](out, param_dict['fc_norm_layer'], training=training,
                                                   num_step=num_step)

        out = F.leaky_relu(out)

        if not self.args.max_pooling:
            out = F.avg_pool2d(out, out.shape[2])

        out = out.view(out.size(0), -1)
        out = self.layer_dict['linear'](out, param_dict['linear'])

        return out

    def restore_backup_stats(self):
        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)].restore_backup_stats()

        if self.args.norm_layer == "batch_norm":
            self.layer_dict['fc_norm_layer'].restore_backup_stats()
