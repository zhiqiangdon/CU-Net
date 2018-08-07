import torch
import numpy
import math
import torch.nn as nn
from options.train_options import TrainOptions


opt = TrainOptions().parse() 
bitsW = opt.bits_w
bitsI = opt.bits_i
bitsG = opt.bits_g

# Scale

def S(bits):
  return 2.0 ** (bits - 1)

# Clip function

def C(x, bits=32):
  if bits > 15 or bits == 1 or bits == 2:
    delta = 0.
  else:
    delta = 1. / S(bits)
  MAX = +1 - delta
  MIN = -1 + delta
  x = torch.clamp(x, MIN, MAX)
  return x


# Quantization function Q(x, k)

def Q(x, bits):
  if bits > 15:
    return x
  elif bits == 1:
    return torch.sign(x)
  elif bits == 2:  
    return torch.round(x)
  else:
    SCALE = S(bits)
    return torch.round(x * SCALE) / SCALE


### QuanInput2d ###

class QuanInput(torch.autograd.Function):
    '''
    Quantize the input activations and calculate the mean across channel dimension.
    '''
    # @staticmethod
    def forward(self, x):
        self.save_for_backward(x)
        x = Q(C(x, bitsI), bitsI)
        return x

    # @staticmethod
    def backward(self, grad_output):
        x, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.ge(1)] = 0
        grad_input[x.le(-1)] = 0
        return grad_input


class QuanInput2d(nn.Module):
    def __init__(self):
        super(QuanInput2d, self).__init__()
        self.layer_type = 'QuanInput2d'
    
    def forward(self, x):
        x = QuanInput()(x)
        return x

### QuanOp() ###

class QuanOp():
    def __init__(self, model):
        # count the number of Conv2d
        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                count_Conv2d = count_Conv2d + 1

        start_range = 1
        end_range = count_Conv2d - 2  # leave out the first and the last conv2d
        
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def quantization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.quantizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = C(self.target_modules[index].data, bitsG)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(Q(self.target_modules[index].data, bitsG))

    def quantizeConvParams(self):
        for index in range(self.num_of_params):
            if bitsW == 1:
              n = self.target_modules[index].data[0].nelement()
              s = self.target_modules[index].data.size()
              m = self.target_modules[index].data.norm(1, 3)\
                      .sum(2).sum(1).div(n).expand(s)
              m = Q(m, bitsG)     
              self.target_modules[index].data.sign()\
                      .mul(m, out=self.target_modules[index].data)
            if bitsW == 2:
              w = self.target_modules[index].data
              n = self.target_modules[index].data[0].nelement()
              s = self.target_modules[index].data.size()
              d = self.target_modules[index].data.norm(1, 3)\
                      .sum(2).sum(1).div(n).mul(0.7)
              wt = w
              for col in range(s[0]):
                  d_col = d[col,0,0,0]
                  wt_neg = w[col,:,:,:].lt(-1.0 * d_col).float().mul(-1)
                  wt_pos = w[col,:,:,:].gt(1.0  * d_col).float()
                  wt[col,:,:,:] = wt_pos.add(wt_neg)
              wt.mul(1, out=self.target_modules[index].data)        
            else:
              self.target_modules[index].data = Q(C(self.target_modules[index].data, bitsW), bitsW)

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    # for gradient
    def updateQuanGradWeight(self):
        for index in range(self.num_of_params):
          if bitsW == 1:
              weight = self.target_modules[index].data
              n = weight[0].nelement()
              s = weight.size()
              m = weight.norm(1, 3)\
                      .sum(2).sum(1).div(n).expand(s)
              m[weight.lt(-1.0)] = 0 
              m[weight.gt(1.0)] = 0
              m = Q(m, bitsG)
              m = m.mul(self.target_modules[index].grad.data)
              m_add = weight.sign().mul(self.target_modules[index].grad.data)
              m_add = m_add.sum(3)\
                      .sum(2).sum(1).div(n).expand(s)
              m_add = m_add.mul(weight.sign())
              self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)
              self.target_modules[index].grad.data = Q(C(self.target_modules[index].grad.data, bitsG), bitsG)
          else:
              self.target_modules[index].grad.data = Q(C(self.target_modules[index].grad.data, bitsG), bitsG)

