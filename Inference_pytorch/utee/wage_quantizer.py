import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Function

def shift(x):
    #TODO: edge case, when x contains 0
    return 2.**torch.round(torch.log2(x))

def S(bits):
    return 2.**(bits-1)

def SR(x):
    r = torch.cuda.FloatTensor(*x.size()).uniform_()
    return torch.floor(x+r)

def C(x, bits):
    if bits > 15 or bits == 1:
        delta = 0
    else:
        delta = 1. / S(bits)
    upper = 1  - delta
    lower = -1 + delta
    return torch.clamp(x, lower, upper)

def Q(x, bits):
    assert bits != -1
    if bits==1:
        return torch.sign(x)
    if bits > 15:
        return x
    return torch.round(x*S(bits))/S(bits)

def QW(x, bits, scale=1.0):
    y = Q(C(x, bits), bits)
    # per layer scaling
    if scale>1.8: y /= scale
    return y

def QE(x, bits):
    max_entry = x.abs().max()
    assert max_entry != 0, "QE blow"
    #if max_entry != 0:
    x /= shift(max_entry)
    return Q(C(x, bits), bits)

def QG(x, bits_G, lr):
    max_entry = x.abs().max()
    assert max_entry != 0, "QG blow"
    #if max_entry != 0:
    x /= shift(max_entry)
    norm = lr * x
    norm = SR(norm)
    return norm / S(bits_G)
    #else:
    #    return x

class WAGERounding(Function):
    @staticmethod
    def forward(self, x, bits_A, bits_E, optional):
        self.optional = optional
        self.bits_E = bits_E
        self.save_for_backward(x)
        if bits_A == -1: ret = x
        else: ret = Q(x, bits_A)

        return ret

    @staticmethod
    def backward(self, grad_output):
        if self.bits_E == -1: return grad_output, None, None, None

        if self.needs_input_grad[0]:
            try:
                grad_input = QE(grad_output, self.bits_E)
            except AssertionError as e:
                print("="*80)
                print("Error backward:%s"%self.optional)
                print("-"*80)
                print(grad_output.max())
                print(grad_output.min())
                print("="*80)
                raise e
        else:
            grad_input = grad_output

        return grad_input, None, None, None

class WAGERounding_forward(Function):
    @staticmethod
    def forward(self, x, bits_A, bits_E, optional):
        self.optional = optional
        self.bits_E = bits_E
        self.save_for_backward(x)
        if bits_A == -1: ret = x
        else: ret = Q(x, bits_A)

        return ret

    @staticmethod
    def backward(self, grad_output):
        return grad_output, None, None, None


quantize_wage = WAGERounding.apply

class WAGEQuantizer(Module):
    def __init__(self, bits_A, bits_E, name="", writer=None):
        super(WAGEQuantizer, self).__init__()
        self.bits_A = bits_A
        self.bits_E = bits_E
        self.name = name
        self.writer = writer

    def forward(self, x):
        if self.bits_A != -1:
            x = C(x, self.bits_A) #  keeps the gradients
        #print(x.std())
        y = quantize_wage(x, self.bits_A, self.bits_E, self.name)
        if self.writer is not None:
            self.writer.add_histogram(
                    "activation-before/%s"%self.name, x.clone().cpu().data.numpy())
            self.writer.add_histogram(
                    "activation-after/%s"%self.name, y.clone().cpu().data.numpy())
        return y

def WAGEQuantizer_f(x, bits_A, bits_E, name=""):
        if bits_A != -1:
            x = C(x, bits_A) #  keeps the gradients
        y = quantize_wage(x, bits_A, bits_E, name)
        return y

if __name__ == "__main__":
    import numpy as np
    np.random.seed(10)
    shape = (5,5)
    # test QG
    test_data = np.random.rand(*shape)
    r = np.random.rand(*shape)
    print(test_data*10)
    print(r*10)
    test_tensor = torch.from_numpy(test_data).float()
    rand_tensor = torch.from_numpy(r).float()
    lr = 2
    bits_W = 2
    bits_G = 8
    bits_A = 8
    bits_E = 8
    bits_R = 16
    print("="*80)
    print("Gradient")
    print("="*80)
    quant_data = QG(test_tensor, bits_G, bits_R, lr, rand_tensor).data.numpy()
    print(quant_data)
    # test QA
    print("="*80)
    print("Activation")
    print("="*80)
    quant_data = QA(test_tensor, bits_A).data.numpy()
    print(quant_data)
    # test QW
    print("="*80)
    print("Weight")
    print("="*80)
    quant_data = QW(test_tensor, bits_W, scale=16.0).data.numpy()
    print(quant_data)
    # test QW
    print("="*80)
    print("Error")
    print("="*80)
    quant_data = QE(test_tensor, bits_E).data.numpy()
    print(quant_data)

