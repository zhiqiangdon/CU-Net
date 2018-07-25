# Zhiqiang Tang, Feb 2017
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

class _Bn_Relu_Conv1x1(nn.Sequential):
    def __init__(self, in_num, out_num):
        super(_Bn_Relu_Conv1x1, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_num))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_num, out_num, kernel_size=1,
                                          stride=1, bias=False))

class _Bn_Relu_Conv3x3(nn.Sequential):
    def __init__(self, in_num, out_num):
        super(_Bn_Relu_Conv3x3, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_num))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_num, out_num, kernel_size=3,
                                          stride=1, padding=1, bias=False))

class _Adapter(nn.Module):
    def __init__(self, num_input_features, num_output_features, efficient):
        super(_Adapter, self).__init__()
        self.add_module('adapter_norm', nn.BatchNorm2d(num_input_features))
        self.add_module('adapter_relu', nn.ReLU(inplace=True))
        self.add_module('adapter_conv', nn.Conv2d(num_input_features, num_output_features,
                                                  kernel_size=1, stride=1, bias=False))
        self.efficient = efficient

    def forward(self, prev_features):
        bn_function = _bn_function_factory(self.adapter_norm, self.adapter_relu,
                                           self.adapter_conv)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            adapter_output = cp.checkpoint(bn_function, *prev_features)
        else:
            adapter_output = bn_function(*prev_features)

        return adapter_output

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate,
                 bn_size, drop_rate, efficient=True):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.Module):
    def __init__(self, layer_num, in_num, neck_size, growth_rate,
                 drop_rate=0, efficient=True, requires_skip=True, is_up=False):
        self.layer_num = layer_num
        self.requires_skip = requires_skip
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        print('layer number is %d' % layer_num)
        for i in range(0, layer_num):
            tmp_in_num = in_num + i * growth_rate
            print('layer %d input channel number is %d' % (i, tmp_in_num))
            self.layers.append(_DenseLayer(tmp_in_num, growth_rate=growth_rate,
                                           bn_size=neck_size, drop_rate=drop_rate,
                                           efficient=efficient))

        # self.adapters_ahead = nn.ModuleList()
        # adapter_in_nums = []
        adapter_in_num = in_num + layer_num * growth_rate
        adapter_out_num = in_num
        if is_up:
            adapter_out_num = adapter_out_num / 2
        # for i in range(0, layer_num):
        #     tmp_in_num = in_num + (i + 1) * growth_rate
        #     adapter_in_nums.append(tmp_in_num)
        print('adapter input channel number is %d' % adapter_in_num)
        self.adapter_ahead = _Adapter(adapter_in_num, adapter_out_num,
                                      efficient=efficient)
        # self.adapters_ahead = nn.ModuleList(self.adapters_ahead)
        print('adapter output channel number is %d' % adapter_out_num)

        if requires_skip:
            print('creating skip layers ...')
            # self.adapters_skip = nn.ModuleList()
            # for i in range(0, layer_num):
            self.adapter_skip = _Adapter(adapter_in_num, adapter_out_num,
                                         efficient=efficient)

    def forward(self, x):

        # self.saved_features = []
        if type(x) is torch.Tensor:
            x = [x]
        if type(x) is not list:
            raise Exception('type(x) should be list, but it is: ', type(x))
        # for t_x in x:
        #     print 't_x type: ', type(t_x)
        #     print 't_x size: ', t_x.size()
        for i in range(0, self.layer_num):
            out = self.layers[i](x)
            x.append(out)
        # x = x + self.saved_features
        # for t_x in x:
        #     print 't_x type: ', type(t_x)
        #     print 't_x size: ', t_x.size()

        out_ahead = self.adapter_ahead(x)
        if self.requires_skip:
            out_skip = self.adapter_skip(x)
            return out_ahead, out_skip
        else:
            return out_ahead

class _TransitionDown(nn.Module):
    def __init__(self, in_num, out_num):
        super(_TransitionDown, self).__init__()
        self.adapter = _Bn_Relu_Conv1x1(in_num=in_num, out_num=out_num)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.adapter(x)
        out = self.pool(x)
        return out

class _TransitionUp(nn.Sequential):
    def __init__(self, in_num, out_num):
        super(_TransitionUp, self).__init__()
        self.adapter = _Bn_Relu_Conv1x1(in_num=in_num, out_num=out_num)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.adapter(x)
        out = self.upsample(x)
        return out

class _Hourglass(nn.Module):
    def __init__(self, layer_num, in_num, neck_size, growth_rate):
        super(_Hourglass, self).__init__()
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.num_blocks = 4
        print('creating hg ...')
        for i in range(0, self.num_blocks):
            print('creating down block %d ...' % i)
            self.down_blocks.append(_DenseBlock(layer_num=layer_num, in_num=in_num,
                                                neck_size=neck_size, growth_rate=growth_rate,
                                                requires_skip=True))
            print('creating up block %d ...' % i)
            self.up_blocks.append(_DenseBlock(in_num=in_num * 2, neck_size=neck_size,
                                              growth_rate=growth_rate, layer_num=layer_num,
                                              requires_skip=False, is_up=True))
        # self.down_blocks = nn.ModuleList(self.down_blocks)
        # self.up_blocks = nn.ModuleList(self.up_blocks)
        print('creating neck block ...')
        self.neck_block = _DenseBlock(in_num=in_num, neck_size=neck_size,
                                      growth_rate=growth_rate, layer_num=layer_num,
                                      requires_skip=False)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)


    def forward(self, x):
        skip_list = [None] * self.num_blocks
        # print 'input x size is ', x.size()
        for j in range(0, self.num_blocks):
            # print('using down block %d ...' % j)
            x, skip_list[j] = self.down_blocks[j](x)
            # print 'output size is ', x.size()
            # print 'skip size is ', skip_list[j].size()
            x = self.maxpool(x)
        # print('using neck block ...')
        x = self.neck_block(x)
        # print 'output size is ', x.size()
        for j in list(reversed(range(0, self.num_blocks))):
            x = self.upsample(x)
            # print('using up block %d ...' % j)
            x = self.up_blocks[j]([x, skip_list[j]])
            # print 'output size is ', x.size()
        return x

class _HourglassWrapper(nn.Module):
    def __init__(self, layer_num, neck_size, growth_rate, init_chan_num, num_classes):

        print('layer number is %d' % layer_num)
        print('growth rate is %d' % growth_rate)
        print('neck size is %d' % neck_size)
        print('class number is %d' % num_classes)
        print('initial channel number is %d' % init_chan_num)
        super(_HourglassWrapper, self).__init__()
        self.features0 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, init_chan_num, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(init_chan_num)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        num_chans = init_chan_num
        print('channel number is %d' % num_chans)
        self.hg = _Hourglass(layer_num=layer_num, in_num=num_chans,
                              neck_size=neck_size, growth_rate=growth_rate)

        self.linear = _Bn_Relu_Conv1x1(in_num=num_chans, out_num=num_classes)


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
        # out = self.features(x)
        x = self.features0(x)
        x = self.hg(x)
        out = self.linear(x)
        return out

def create_dense_unet(layer_num, neck_size, growth_rate,
                      init_chan_num, num_classes):

    net = _HourglassWrapper(layer_num, neck_size, growth_rate,
                            init_chan_num, num_classes)
    return net


