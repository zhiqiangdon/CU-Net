# Zhiqiang Tang, Feb 2017
import torch
import torch.nn as nn
import math
from collections import OrderedDict
from torch.autograd import Variable, Function
from torch._thnn import type2backend
from torch.backends import cudnn
from functools import reduce
from operator import mul
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
from options.train_options import TrainOptions
from utils.quantize import *

opt = TrainOptions().parse() 
bitsI = opt.bits_i

class _SharedAllocation(object):
    """
    A helper class which maintains a shared memory allocation.
    Used for concatenation and batch normalization.
    """
    def __init__(self, storage):
        self.storage = storage

    def type(self, t):
        self.storage = self.storage.type(t)

    def type_as(self, obj):
        if isinstance(obj, Variable):
            self.storage = self.storage.type(obj.data.storage().type())
        elif isinstance(obj, torch._TensorBase):
            self.storage = self.storage.type(obj.storage().type())
        else:
            self.storage = self.storage.type(obj.type())

    def resize_(self, size):
        if self.storage.size() < size:
            self.storage.resize_(size)
        return self

class _EfficientDensenetBottleneck(nn.Module):
    """
    A optimized layer which encapsulates the batch normalization, ReLU, and
    convolution operations within the bottleneck of a DenseNet layer.
    This layer usage shared memory allocations to store the outputs of the
    concatenation and batch normalization features. Because the shared memory
    is not perminant, these features are recomputed during the backward pass.
    """
    def __init__(self, shared_allocation_1, shared_allocation_2, num_input_channels, num_output_channels):

        super(_EfficientDensenetBottleneck, self).__init__()
        self.shared_allocation_1 = shared_allocation_1
        self.shared_allocation_2 = shared_allocation_2
        self.num_input_channels = num_input_channels

        self.norm_weight = nn.Parameter(torch.Tensor(num_input_channels))
        self.norm_bias = nn.Parameter(torch.Tensor(num_input_channels))
        self.register_buffer('norm_running_mean', torch.zeros(num_input_channels))
        self.register_buffer('norm_running_var', torch.ones(num_input_channels))
        self.conv_weight = nn.Parameter(torch.Tensor(num_output_channels, num_input_channels, 1, 1))
        self._reset_parameters()


    def _reset_parameters(self):
        self.norm_running_mean.zero_()
        self.norm_running_var.fill_(1)
        self.norm_weight.data.uniform_()
        self.norm_bias.data.zero_()
        stdv = 1. / math.sqrt(self.num_input_channels)
        self.conv_weight.data.uniform_(-stdv, stdv)


    def forward(self, inputs):
        if isinstance(inputs, Variable):
            inputs = [inputs]
        fn = _EfficientDensenetBottleneckFn(self.shared_allocation_1, self.shared_allocation_2,
            self.norm_running_mean, self.norm_running_var,
            stride=1, padding=0, dilation=1, groups=1,
            training=self.training, momentum=0.1, eps=1e-5)
        return fn(self.norm_weight, self.norm_bias, self.conv_weight, *inputs)

class _DenseLayer(nn.Sequential):

    def __init__(self, shared_allocation_1, shared_allocation_2, in_num, neck_size, growth_rate):
        super(_DenseLayer, self).__init__()
        self.shared_allocation_1 = shared_allocation_1
        self.shared_allocation_2 = shared_allocation_2

        self.add_module('bottleneck', _EfficientDensenetBottleneck(shared_allocation_1, shared_allocation_2,
                                                           in_num, neck_size * growth_rate))
        self.add_module('norm.2', nn.BatchNorm2d(neck_size * growth_rate))
        self.add_module('relu.2', nn.ReLU(inplace=True))
        # QuanInput2d: No.1 
        if bitsI <= 15:
            self.add_module('quaninput.2', QuanInput2d())
        self.add_module('conv.2', nn.Conv2d(neck_size * growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        if isinstance(x, Variable):
            prev_features = [x]
        else:
            prev_features = x
        # print(len(prev_features))
        new_features = super(_DenseLayer, self).forward(prev_features)

        return new_features

class _DenseBlock(nn.Module):
    def __init__(self, in_num, neck_size, growth_rate, layer_num, max_link,
                 storage_size=1024, requires_skip=True, is_up=False):
        input_storage_1 = torch.Storage(storage_size)
        input_storage_2 = torch.Storage(storage_size)
        self.shared_allocation_1 = _SharedAllocation(input_storage_1)
        self.shared_allocation_2 = _SharedAllocation(input_storage_2)
        self.saved_features = []
        self.max_link = max_link
        self.requires_skip = requires_skip
        super(_DenseBlock, self).__init__()
        max_in_num = in_num + max_link * growth_rate
        self.final_num_features = max_in_num
        self.layers = []
        print('layer number is %d' % layer_num)
        for i in range(0, layer_num):
            if i < max_link:
                tmp_in_num = in_num + i * growth_rate
            else:
                tmp_in_num = max_in_num
            print('layer %d input channel number is %d' % (i, tmp_in_num))
            self.layers.append(_DenseLayer(self.shared_allocation_1, self.shared_allocation_2,
                                           tmp_in_num, neck_size, growth_rate))
        self.layers = nn.ModuleList(self.layers)
        self.adapters_ahead = []
        adapter_in_nums = []
        adapter_out_num = in_num
        if is_up:
            adapter_out_num = adapter_out_num / 2
        for i in range(0, layer_num):
            if i < max_link:
                tmp_in_num = in_num + (i+1) * growth_rate
            else:
                tmp_in_num = max_in_num + growth_rate
            adapter_in_nums.append(tmp_in_num)
            print('adapter %d input channel number is %d' % (i, adapter_in_nums[i]))
            self.adapters_ahead.append(_EfficientDensenetBottleneck(self.shared_allocation_1,
                                                                    self.shared_allocation_2,
                                                                    adapter_in_nums[i], adapter_out_num))
        self.adapters_ahead = nn.ModuleList(self.adapters_ahead)
        print('adapter output channel number is %d' % adapter_out_num)
        if requires_skip:
            print('creating skip layers ...')
            self.adapters_skip = []
            for i in range(0, layer_num):
                self.adapters_skip.append(_EfficientDensenetBottleneck(self.shared_allocation_1,
                                                                       self.shared_allocation_2,
                                                                       adapter_in_nums[i], adapter_out_num))
            self.adapters_skip = nn.ModuleList(self.adapters_skip)

    def forward(self, x, i):
        if i == 0:
            self.saved_features = []
            if isinstance(x, Variable):
                # Update storage type
                self.shared_allocation_1.type_as(x)
                self.shared_allocation_2.type_as(x)
                # Resize storage
                final_size = list(x.size())
            elif isinstance(x, list):
                self.shared_allocation_1.type_as(x[0])
                self.shared_allocation_2.type_as(x[0])
                # Resize storage
                final_size = list(x[0].size())
            else:
                print('invalid type in the input of _DenseBlock module. exiting ...')
                exit()
            # print(final_size)
            final_size[1] = self.final_num_features
            # print(final_size)
            final_storage_size = reduce(mul, final_size, 1)
            # print(final_storage_size)
            self.shared_allocation_1.resize_(final_storage_size)
            self.shared_allocation_2.resize_(final_storage_size)

        if isinstance(x, Variable):
            x = [x]
        x = x + self.saved_features
        out = self.layers[i](x)
        if i < self.max_link:
            self.saved_features.append(out)
        elif len(self.saved_features) != 0:
            self.saved_features.pop(0)
            self.saved_features.append(out)
        x.append(out)
        out_ahead = self.adapters_ahead[i](x)
        if self.requires_skip:
            out_skip = self.adapters_skip[i](x)
            return out_ahead, out_skip
        else:
            return out_ahead

class _IntermediaBlock(nn.Module):
    def __init__(self, in_num, out_num, layer_num, max_link, storage_size=1024):
        input_storage_1 = torch.Storage(storage_size)
        input_storage_2 = torch.Storage(storage_size)
        self.shared_allocation_1 = _SharedAllocation(input_storage_1)
        self.shared_allocation_2 = _SharedAllocation(input_storage_2)
        max_in_num = in_num + out_num * max_link
        self.final_num_features = max_in_num
        self.saved_features = []
        self.max_link = max_link
        super(_IntermediaBlock, self).__init__()
        print('creating intermedia block ...')
        self.adapters = []
        for i in range(0, layer_num-1):
            if i < max_link:
                tmp_in_num = in_num + (i+1) * out_num
            else:
                tmp_in_num = max_in_num
            print('intermedia layer %d input channel number is %d' % (i, tmp_in_num))
            self.adapters.append(_EfficientDensenetBottleneck(self.shared_allocation_1,
                                                              self.shared_allocation_2,
                                                              tmp_in_num, out_num))
        self.adapters = nn.ModuleList(self.adapters)
        print('intermedia layer output channel number is %d' % out_num)

    def forward(self, x, i):
        if i == 0:
            self.saved_features = []
            if isinstance(x, Variable):
                # Update storage type
                self.shared_allocation_1.type_as(x)
                self.shared_allocation_2.type_as(x)
                # Resize storage
                final_size = list(x.size())
                if self.max_link != 0:
                    self.saved_features.append(x)
            elif isinstance(x, list):
                self.shared_allocation_1.type_as(x[0])
                self.shared_allocation_2.type_as(x[0])
                # Resize storage
                final_size = list(x[0].size())
                if self.max_link != 0:
                    self.saved_features = self.saved_features + x
            else:
                print('invalid type in the input of _DenseBlock module. exiting ...')
                exit()
            final_size[1] = self.final_num_features
            # print 'final size of intermedia block is ', final_size
            final_storage_size = reduce(mul, final_size, 1)
            # print(final_storage_size)
            self.shared_allocation_1.resize_(final_storage_size)
            self.shared_allocation_2.resize_(final_storage_size)
            # print('middle list length is %d' % len(self.saved_features))
            return x

        if isinstance(x, Variable):
            # self.saved_features.append(x)
            x = [x]
        x = x + self.saved_features
        out = self.adapters[i-1](x)
        if i < self.max_link:
            self.saved_features.append(out)
        elif len(self.saved_features) != 0:
            self.saved_features.pop(0)
            self.saved_features.append(out)
        # print('middle list length is %d' % len(self.saved_features))
        return out

class _Bn_Relu_Conv1x1(nn.Sequential):
    def __init__(self, in_num, out_num):
        super(_Bn_Relu_Conv1x1, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_num))
        self.add_module('relu', nn.ReLU(inplace=True))
        # QuanInput2d: No.2 
        if bitsI <= 15:
            self.add_module('quaninput', QuanInput2d())
        self.add_module('conv', nn.Conv2d(in_num, out_num,
                                          kernel_size=1, stride=1, bias=False))

# class _TransitionDown(nn.Module):
#     def __init__(self, in_num_list, out_num, num_units):
#         super(_TransitionDown, self).__init__()
#         self.adapters = []
#         for i in range(0, num_units):
#             self.adapters.append(_Bn_Relu_Conv1x1(in_num=in_num_list[i], out_num=out_num))
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#     def forward(self, x, i):
#         x = self.adapters[i](x)
#         out = self.pool(x)
#         return out
#
# class _TransitionUp(nn.Module):
#     def __init__(self, in_num_list, out_num_list, num_units):
#         super(_TransitionUp, self).__init__()
#         self.adapters = []
#         for i in range(0, num_units):
#             self.adapters.append(_Bn_Relu_Conv1x1(in_num=in_num_list[i], out_num=out_num_list[i]))
#         self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
#
#     def forward(self, x, i):
#         x = self.adapters[i](x)
#         out = self.upsample(x)
#         return out


class _Hourglass(nn.Module):
    def __init__(self, in_num, neck_size, growth_rate, layer_num, max_link):
        super(_Hourglass, self).__init__()
        self.down_blocks = []
        self.up_blocks = []
        self.num_blocks = 4
        print('creating hg ...')
        for i in range(0, self.num_blocks):
            print('creating down block %d ...' % i)
            self.down_blocks.append(_DenseBlock(in_num=in_num, neck_size=neck_size,
                                      growth_rate=growth_rate, layer_num=layer_num,
                                      max_link=max_link, requires_skip=True))
            print('creating up block %d ...' % i)
            self.up_blocks.append(_DenseBlock(in_num=in_num*2, neck_size=neck_size,
                                      growth_rate=growth_rate, layer_num=layer_num,
                                      max_link=max_link, requires_skip=False, is_up=True))
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)
        print('creating neck block ...')
        self.neck_block = _DenseBlock(in_num=in_num, neck_size=neck_size,
                                     growth_rate=growth_rate, layer_num=layer_num,
                                     max_link=max_link, requires_skip=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x, i):
        skip_list = [None] * self.num_blocks
        # print 'input x size is ', x.size()
        for j in range(0, self.num_blocks):
            # print('using down block %d ...' % j)
            x, skip_list[j] = self.down_blocks[j](x, i)
            # print 'x size is ', x.size()
            # print 'skip size is ', skip_list[j].size()
            x = self.maxpool(x)
        # print('using neck block ...')
        x = self.neck_block(x, i)
        # print 'output size is ', x.size()
        for j in list(reversed(range(0, self.num_blocks))):
            x = self.upsample(x)
            # print('using up block %d ...' % j)
            x = self.up_blocks[j]([x, skip_list[j]], i)
            # print 'output size is ', x.size()
        return x

class _HourglassWrapper(nn.Module):
    def __init__(self, init_chan_num, neck_size, growth_rate,
                 num_classes, layer_num, max_link, inter_loss_num):
        assert inter_loss_num <= layer_num
        inter_loss_every = float(layer_num) / float(inter_loss_num)
        self.loss_achors = []
        for i in range(0, inter_loss_num):
            tmp_achor = int(round(inter_loss_every * (i + 1)))
            if tmp_achor <= layer_num:
                self.loss_achors.append(tmp_achor)

        assert layer_num in self.loss_achors
        assert inter_loss_num == len(self.loss_achors)

        if max_link >= layer_num:
            print 'max link number is larger than the layer number.'
            exit()
        print('layer number is %d' % layer_num)
        print('loss number is %d' % inter_loss_num)
        print('loss achors are: ', self.loss_achors)
        print('max link number is %d' % max_link)
        print('growth rate is %d' % growth_rate)
        print('neck size is %d' % neck_size)
        print('class number is %d' % num_classes)
        print('initial channel number is %d' % init_chan_num)
        num_chans = init_chan_num
        super(_HourglassWrapper, self).__init__()
        self.layer_num = layer_num
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, init_chan_num, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(init_chan_num)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))
        # self.denseblock0 = _DenseBlock(layer_num=4, in_num=init_chan_num,
        #                                neck_size=neck_size, growth_rate=growth_rate)
        # hg_in_num = init_chan_num + growth_rate * 4
        print('channel number is %d' % num_chans)
        self.hg = _Hourglass(in_num=num_chans, neck_size=neck_size, growth_rate=growth_rate,
                             layer_num=layer_num, max_link=max_link)

        self.linears = []
        for i in range(0, layer_num):
            self.linears.append(_Bn_Relu_Conv1x1(in_num=num_chans, out_num=num_classes))
        self.linears = nn.ModuleList(self.linears)
        # intermedia_in_nums = []
        # for i in range(0, num_units-1):
        #     intermedia_in_nums.append(num_chans * (i+2))
        self.intermedia = _IntermediaBlock(in_num=num_chans, out_num=num_chans,
                                           layer_num=layer_num, max_link=max_link)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                stdv = 1/math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv)
                # m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)
                    # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                # m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, x):
        # print(x.size())
        x = self.features(x)
        # print(x.size())
        # x = self.denseblock0(x)
        # print 'x size is', x.size()
        out = []
        # middle = []
        # middle.append(x)
        for i in range(0, self.layer_num):
            # print('using intermedia layer %d ...' % i)
            x = self.intermedia(x, i)
            # print 'x size after intermedia layer is ', x.size()
            # print('using hg %d ...' % i)
            x = self.hg(x, i)
            # print 'x size after hg is ', x.size()
            # middle.append(x)
            if (i + 1) in self.loss_achors:
                tmp_out = self.linears[i](x)
                # print 'tmp output size is ', tmp_out.size()
                out.append(tmp_out)
            # if i < self.num_units-1:
        # exit()
        assert len(self.loss_achors) == len(out)
        return out

def create_dense_unet(neck_size, growth_rate, init_chan_num,
                      num_classes, layer_num, max_link, inter_loss_num):
    net = _HourglassWrapper(init_chan_num=init_chan_num, neck_size=neck_size,
                            growth_rate=growth_rate, num_classes=num_classes,
                            layer_num=layer_num, max_link=max_link,
                            inter_loss_num=inter_loss_num)
    return net

class _EfficientDensenetBottleneckFn(Function):
    """
    The autograd function which performs the efficient bottlenck operations.
    Each of the sub-operations -- concatenation, batch normalization, ReLU,
    and convolution -- are abstracted into their own classes
    """
    def __init__(self, shared_allocation_1, shared_allocation_2,
            running_mean, running_var,
            stride=1, padding=0, dilation=1, groups=1,
            training=False, momentum=0.1, eps=1e-5):

        self.efficient_cat = _EfficientCat(shared_allocation_1.storage)
        self.efficient_batch_norm = _EfficientBatchNorm(shared_allocation_2.storage, running_mean, running_var,
                training, momentum, eps)
        self.efficient_relu = _EfficientReLU()
        self.efficient_conv = _EfficientConv2d(stride, padding, dilation, groups)

        # Buffers to store old versions of bn statistics
        self.prev_running_mean = self.efficient_batch_norm.running_mean.new()
        self.prev_running_mean.resize_as_(self.efficient_batch_norm.running_mean)
        self.prev_running_var = self.efficient_batch_norm.running_var.new()
        self.prev_running_var.resize_as_(self.efficient_batch_norm.running_var)
        self.curr_running_mean = self.efficient_batch_norm.running_mean.new()
        self.curr_running_mean.resize_as_(self.efficient_batch_norm.running_mean)
        self.curr_running_var = self.efficient_batch_norm.running_var.new()
        self.curr_running_var.resize_as_(self.efficient_batch_norm.running_var)


    def forward(self, bn_weight, bn_bias, conv_weight, *inputs):
        self.prev_running_mean.copy_(self.efficient_batch_norm.running_mean)
        self.prev_running_var.copy_(self.efficient_batch_norm.running_var)

        bn_input = self.efficient_cat.forward(*inputs)
        bn_output = self.efficient_batch_norm.forward(bn_weight, bn_bias, bn_input)
        relu_output = self.efficient_relu.forward(bn_output)
        conv_output = self.efficient_conv.forward(conv_weight, None, relu_output)

        self.bn_weight = bn_weight
        self.bn_bias = bn_bias
        self.conv_weight = conv_weight
        self.inputs = inputs
        return conv_output


    def backward(self, grad_output):
        # Turn off bn training status, and temporarily reset statistics
        training = self.efficient_batch_norm.training
        self.curr_running_mean.copy_(self.efficient_batch_norm.running_mean)
        self.curr_running_var.copy_(self.efficient_batch_norm.running_var)
        # self.efficient_batch_norm.training = False
        self.efficient_batch_norm.running_mean.copy_(self.prev_running_mean)
        self.efficient_batch_norm.running_var.copy_(self.prev_running_var)

        # Recompute concat and BN
        cat_output = self.efficient_cat.forward(*self.inputs)
        bn_output = self.efficient_batch_norm.forward(self.bn_weight, self.bn_bias, cat_output)
        relu_output = self.efficient_relu.forward(bn_output)

        # Conv backward
        conv_weight_grad, _, conv_grad_output = self.efficient_conv.backward(
                self.conv_weight, None, relu_output, grad_output)

        # ReLU backward
        relu_grad_output = self.efficient_relu.backward(bn_output, conv_grad_output)

        # BN backward
        self.efficient_batch_norm.running_mean.copy_(self.curr_running_mean)
        self.efficient_batch_norm.running_var.copy_(self.curr_running_var)
        bn_weight_grad, bn_bias_grad, bn_grad_output = self.efficient_batch_norm.backward(
                self.bn_weight, self.bn_bias, cat_output, relu_grad_output)

        # Input backward
        grad_inputs = self.efficient_cat.backward(bn_grad_output)

        # Reset bn training status and statistics
        self.efficient_batch_norm.training = training
        self.efficient_batch_norm.running_mean.copy_(self.curr_running_mean)
        self.efficient_batch_norm.running_var.copy_(self.curr_running_var)

        return tuple([bn_weight_grad, bn_bias_grad, conv_weight_grad] + list(grad_inputs))


# The following helper classes are written similarly to pytorch autogrd functions.
# However, they are designed to work on tensors, not variables, and therefore
# are not functions.


class _EfficientBatchNorm(object):
    def __init__(self, storage, running_mean, running_var,
            training=False, momentum=0.1, eps=1e-5):
        self.storage = storage
        self.running_mean = running_mean
        self.running_var = running_var
        self.training = training
        self.momentum = momentum
        self.eps = eps

    def forward(self, weight, bias, input):
        # Assert we're using cudnn
        for i in ([weight, bias, input]):
            if i is not None and not(cudnn.is_acceptable(i)):
                raise Exception('You must be using CUDNN to use _EfficientBatchNorm')

        # Create save variables
        self.save_mean = self.running_mean.new()
        self.save_mean.resize_as_(self.running_mean)
        self.save_var = self.running_var.new()
        self.save_var.resize_as_(self.running_var)

        # Do forward pass - store in input variable
        res = type(input)(self.storage)
        res.resize_as_(input)
        torch._C._cudnn_batch_norm_forward(
            input, res, weight, bias, self.running_mean, self.running_var,
            self.save_mean, self.save_var, self.training, self.momentum, self.eps
        )

        return res

    def recompute_forward(self, weight, bias, input):
        # Do forward pass - store in input variable
        res = type(input)(self.storage)
        res.resize_as_(input)
        torch._C._cudnn_batch_norm_forward(
            input, res, weight, bias, self.running_mean, self.running_var,
            self.save_mean, self.save_var, self.training, self.momentum, self.eps
        )

        return res

    def backward(self, weight, bias, input, grad_output):
        # Create grad variables
        grad_weight = weight.new()
        grad_weight.resize_as_(weight)
        grad_bias = bias.new()
        grad_bias.resize_as_(bias)

        # Run backwards pass - result stored in grad_output
        grad_input = grad_output
        torch._C._cudnn_batch_norm_backward(
            input, grad_output, grad_input, grad_weight, grad_bias,
            weight, self.running_mean, self.running_var, self.save_mean,
            self.save_var, self.training, self.eps
        )

        # Unpack grad_output
        res = tuple([grad_weight, grad_bias, grad_input])
        return res


class _EfficientCat(object):
    def __init__(self, storage):
        self.storage = storage

    def forward(self, *inputs):
        # Get size of new varible
        self.all_num_channels = [input.size(1) for input in inputs]
        size = list(inputs[0].size())
        for num_channels in self.all_num_channels[1:]:
            size[1] += num_channels

        # Create variable, using existing storage
        res = type(inputs[0])(self.storage).resize_(size)
        torch.cat(inputs, dim=1, out=res)
        return res

    def backward(self, grad_output):
        # Return a table of tensors pointing to same storage
        res = []
        index = 0
        for num_channels in self.all_num_channels:
            new_index = num_channels + index
            res.append(grad_output[:, index:new_index])
            index = new_index

        return tuple(res)


class _EfficientReLU(object):
    def __init__(self):
        pass

    def forward(self, input):
        backend = type2backend[type(input)]
        output = input
        backend.Threshold_updateOutput(backend.library_state, input, output, 0, 0, True)
        return output

    def backward(self, input, grad_output):
        grad_input = grad_output
        grad_input.masked_fill_(input <= 0, 0)
        return grad_input


class _EfficientConv2d(object):
    def __init__(self, stride=1, padding=0, dilation=1, groups=1):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def _output_size(self, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding
            kernel = self.dilation * (weight.size(d + 2) - 1) + 1
            stride = self.stride
            output_size += ((in_size + (2 * pad) - kernel) // stride + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError("convolution input is too small (output would be {})".format(
                             'x'.join(map(str, output_size))))
        return output_size

    def forward(self, weight, bias, input):
        # Assert we're using cudnn
        for i in ([weight, bias, input]):
            if i is not None and not(cudnn.is_acceptable(i)):
                raise Exception('You must be using CUDNN to use _EfficientBatchNorm')

        res = input.new(*self._output_size(input, weight))
        self._cudnn_info = torch._C._cudnn_convolution_full_forward(
            input, weight, bias, res,
            (self.padding, self.padding),
            (self.stride, self.stride),
            (self.dilation, self.dilation),
            self.groups, cudnn.benchmark
        )

        return res

    def backward(self, weight, bias, input, grad_output):
        grad_input = input.new()
        grad_input.resize_as_(input)
        torch._C._cudnn_convolution_backward_data(
            grad_output, grad_input, weight, self._cudnn_info,
            cudnn.benchmark)

        grad_weight = weight.new().resize_as_(weight)
        torch._C._cudnn_convolution_backward_filter(grad_output, input, grad_weight, self._cudnn_info,
                                                    cudnn.benchmark)

        if bias is not None:
            grad_bias = bias.new().resize_as_(bias)
            torch._C._cudnn_convolution_backward_bias(grad_output, grad_bias, self._cudnn_info)
        else:
            grad_bias = None

        return grad_weight, grad_bias, grad_input


