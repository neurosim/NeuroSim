import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

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
    
def Retention(x, t, v, detect, target):
    lower = torch.min(x).item()
    upper = torch.max(x).item()
    target = (torch.max(x).item() - torch.min(x).item())*target
    if detect == 1: # need to define the sign of v 
        sign = torch.zeros_like(x)
        truncateX = (x+1)/2
        truncateTarget = (target+1)/2
        sign = torch.sign(torch.add(torch.zeros_like(x),truncateTarget)-truncateX)
        ratio = t**(v*sign)
    else :  # random generate target for each cell
        sign = torch.randint_like(x, -1, 2)
        truncateX = (x+1)/2
        ratio = t**(v*sign)

    return torch.clamp((2*truncateX*ratio-1), lower, upper)


def NonLinearQuantizeOut(x, bit):
    minQ = torch.min(x)
    delta = torch.max(x) - torch.min(x)
    #print(minQ)
    #print(delta)
    if (bit == 3) :
        # 3-bit ADC
        y = x.clone()
        base = torch.zeros_like(y)

        bound = np.array([0.02, 0.08, 0.12, 0.18, 0.3, 0.5, 0.7, 1])
        out = np.array([0.01, 0.05, 0.1, 0.15, 0.24, 0.4, 0.6, 0.85])

        ref = torch.from_numpy(bound).float()
        quant = torch.from_numpy(out).float()

        y = torch.where(y<(minQ+ref[0]*delta), torch.add(base,(minQ+quant[0]*delta)), y)
        y = torch.where(((minQ+ref[0]*delta)<=y) & (y<(minQ+ref[1]*delta)), torch.add(base,(minQ+quant[1]*delta)), y)
        y = torch.where(((minQ+ref[1]*delta)<=y) & (y<(minQ+ref[2]*delta)), torch.add(base,(minQ+quant[2]*delta)), y)
        y = torch.where(((minQ+ref[2]*delta)<=y) & (y<(minQ+ref[3]*delta)), torch.add(base,(minQ+quant[3]*delta)), y)
        y = torch.where(((minQ+ref[3]*delta)<=y) & (y<(minQ+ref[4]*delta)), torch.add(base,(minQ+quant[4]*delta)), y)
        y = torch.where(((minQ+ref[4]*delta)<=y) & (y<(minQ+ref[5]*delta)), torch.add(base,(minQ+quant[5]*delta)), y)
        y = torch.where(((minQ+ref[5]*delta)<=y) & (y<(minQ+ref[6]*delta)), torch.add(base,(minQ+quant[6]*delta)), y)
        y = torch.where(((minQ+ref[6]*delta)<=y) & (y<(minQ+ref[7]*delta)), torch.add(base,(minQ+quant[7]*delta)), y)
        
    elif (bit == 4):
        y = x.clone()
        # 4-bit ADC
        base = torch.zeros_like(y)
        
        # good for 2-bit cell
        bound = np.array([0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.85, 1])
        out = np.array([0.01, 0.035, 0.065, 0.1, 0.14, 0.18, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.55, 0.65, 0.775, 0.925])
        
        ref = torch.from_numpy(bound).float()
        quant = torch.from_numpy(out).float()

        y = torch.where(y.data<(minQ+ref[0]*delta), torch.add(base,(minQ+quant[0]*delta)), y)
        y = torch.where(((minQ+ref[0]*delta)<=y.data) & (y.data<(minQ+ref[1]*delta)), torch.add(base,(minQ+quant[1]*delta)), y)
        y = torch.where(((minQ+ref[1]*delta)<=y.data) & (y.data<(minQ+ref[2]*delta)), torch.add(base,(minQ+quant[2]*delta)), y)
        y = torch.where(((minQ+ref[2]*delta)<=y.data) & (y.data<(minQ+ref[3]*delta)), torch.add(base,(minQ+quant[3]*delta)), y)
        y = torch.where(((minQ+ref[3]*delta)<=y.data) & (y.data<(minQ+ref[4]*delta)), torch.add(base,(minQ+quant[4]*delta)), y)
        y = torch.where(((minQ+ref[4]*delta)<=y.data) & (y.data<(minQ+ref[5]*delta)), torch.add(base,(minQ+quant[5]*delta)), y)
        y = torch.where(((minQ+ref[5]*delta)<=y.data) & (y.data<(minQ+ref[6]*delta)), torch.add(base,(minQ+quant[6]*delta)), y)
        y = torch.where(((minQ+ref[6]*delta)<=y.data) & (y.data<(minQ+ref[7]*delta)), torch.add(base,(minQ+quant[7]*delta)), y)
        y = torch.where(((minQ+ref[7]*delta)<=y.data) & (y.data<(minQ+ref[8]*delta)), torch.add(base,(minQ+quant[8]*delta)), y)
        y = torch.where(((minQ+ref[8]*delta)<=y.data) & (y.data<(minQ+ref[9]*delta)), torch.add(base,(minQ+quant[9]*delta)), y)
        y = torch.where(((minQ+ref[9]*delta)<=y.data) & (y.data<(minQ+ref[10]*delta)), torch.add(base,(minQ+quant[10]*delta)), y)
        y = torch.where(((minQ+ref[10]*delta)<=y.data) & (y.data<(minQ+ref[11]*delta)), torch.add(base,(minQ+quant[11]*delta)), y)
        y = torch.where(((minQ+ref[11]*delta)<=y.data) & (y.data<(minQ+ref[12]*delta)), torch.add(base,(minQ+quant[12]*delta)), y)
        y = torch.where(((minQ+ref[12]*delta)<=y.data) & (y.data<(minQ+ref[13]*delta)), torch.add(base,(minQ+quant[13]*delta)), y)
        y = torch.where(((minQ+ref[13]*delta)<=y.data) & (y.data<(minQ+ref[14]*delta)), torch.add(base,(minQ+quant[14]*delta)), y)
        y = torch.where(((minQ+ref[14]*delta)<=y.data) & (y.data<(minQ+ref[15]*delta)), torch.add(base,(minQ+quant[15]*delta)), y)
        
    elif (bit == 5):
        y = x.clone()
        # 5-bit ADC
        base = torch.zeros_like(y)
        """
        # good for 2-bit cell
        bound = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
        out = np.array([0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975])
        """
        
        # 4-bit cell
        bound = np.array([0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1])
        out = np.array([0.001, 0.003, 0.007, 0.010, 0.015, 0.020, 0.030, 0.040, 0.055, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.32, 0.37, 0.42, 0.47, 0.52, 0.57, 0.62, 0.67, 0.75, 0.85, 0.95])
        
        ref = torch.from_numpy(bound).float()
        quant = torch.from_numpy(out).float()

        y = torch.where(y<(minQ+ref[0]*delta), torch.add(base,minQ+quant[0]*delta), y)
        y = torch.where(((minQ+ref[0]*delta)<=y) & (y<(minQ+ref[1]*delta)), torch.add(base,minQ+quant[1]*delta), y)
        y = torch.where(((minQ+ref[1]*delta)<=y) & (y<(minQ+ref[2]*delta)), torch.add(base,minQ+quant[2]*delta), y)
        y = torch.where(((minQ+ref[2]*delta)<=y) & (y<(minQ+ref[3]*delta)), torch.add(base,minQ+quant[3]*delta), y)
        y = torch.where(((minQ+ref[3]*delta)<=y) & (y<(minQ+ref[4]*delta)), torch.add(base,minQ+quant[4]*delta), y)
        y = torch.where(((minQ+ref[4]*delta)<=y) & (y<(minQ+ref[5]*delta)), torch.add(base,minQ+quant[5]*delta), y)
        y = torch.where(((minQ+ref[5]*delta)<=y) & (y<(minQ+ref[6]*delta)), torch.add(base,minQ+quant[6]*delta), y)
        y = torch.where(((minQ+ref[6]*delta)<=y) & (y<(minQ+ref[7]*delta)), torch.add(base,minQ+quant[7]*delta), y)
        y = torch.where(((minQ+ref[7]*delta)<=y) & (y<(minQ+ref[8]*delta)), torch.add(base,minQ+quant[8]*delta), y)
        y = torch.where(((minQ+ref[8]*delta)<=y) & (y<(minQ+ref[9]*delta)), torch.add(base,minQ+quant[9]*delta), y)
        y = torch.where(((minQ+ref[9]*delta)<=y) & (y<(minQ+ref[10]*delta)), torch.add(base,minQ+quant[10]*delta), y)
        y = torch.where(((minQ+ref[10]*delta)<=y) & (y<(minQ+ref[11]*delta)), torch.add(base,minQ+quant[11]*delta), y)
        y = torch.where(((minQ+ref[11]*delta)<=y) & (y<(minQ+ref[12]*delta)), torch.add(base,minQ+quant[12]*delta), y)
        y = torch.where(((minQ+ref[12]*delta)<=y) & (y<(minQ+ref[13]*delta)), torch.add(base,minQ+quant[13]*delta), y)
        y = torch.where(((minQ+ref[13]*delta)<=y) & (y<(minQ+ref[14]*delta)), torch.add(base,minQ+quant[14]*delta), y)
        y = torch.where(((minQ+ref[14]*delta)<=y) & (y<(minQ+ref[15]*delta)), torch.add(base,minQ+quant[15]*delta), y)
        y = torch.where(((minQ+ref[15]*delta)<=y) & (y<(minQ+ref[16]*delta)), torch.add(base,minQ+quant[16]*delta), y)
        y = torch.where(((minQ+ref[16]*delta)<=y) & (y<(minQ+ref[17]*delta)), torch.add(base,minQ+quant[17]*delta), y)
        y = torch.where(((minQ+ref[17]*delta)<=y) & (y<(minQ+ref[18]*delta)), torch.add(base,minQ+quant[18]*delta), y)
        y = torch.where(((minQ+ref[18]*delta)<=y) & (y<(minQ+ref[19]*delta)), torch.add(base,minQ+quant[19]*delta), y)
        y = torch.where(((minQ+ref[19]*delta)<=y) & (y<(minQ+ref[20]*delta)), torch.add(base,minQ+quant[20]*delta), y)
        y = torch.where(((minQ+ref[20]*delta)<=y) & (y<(minQ+ref[21]*delta)), torch.add(base,minQ+quant[21]*delta), y)
        y = torch.where(((minQ+ref[21]*delta)<=y) & (y<(minQ+ref[22]*delta)), torch.add(base,minQ+quant[22]*delta), y)
        y = torch.where(((minQ+ref[22]*delta)<=y) & (y<(minQ+ref[23]*delta)), torch.add(base,minQ+quant[23]*delta), y)
        y = torch.where(((minQ+ref[23]*delta)<=y) & (y<(minQ+ref[24]*delta)), torch.add(base,minQ+quant[24]*delta), y)
        y = torch.where(((minQ+ref[24]*delta)<=y) & (y<(minQ+ref[25]*delta)), torch.add(base,minQ+quant[25]*delta), y)
        y = torch.where(((minQ+ref[25]*delta)<=y) & (y<(minQ+ref[26]*delta)), torch.add(base,minQ+quant[26]*delta), y)
        y = torch.where(((minQ+ref[26]*delta)<=y) & (y<(minQ+ref[27]*delta)), torch.add(base,minQ+quant[27]*delta), y)
        y = torch.where(((minQ+ref[27]*delta)<=y) & (y<(minQ+ref[28]*delta)), torch.add(base,minQ+quant[28]*delta), y)
        y = torch.where(((minQ+ref[28]*delta)<=y) & (y<(minQ+ref[29]*delta)), torch.add(base,minQ+quant[29]*delta), y)
        y = torch.where(((minQ+ref[29]*delta)<=y) & (y<(minQ+ref[30]*delta)), torch.add(base,minQ+quant[30]*delta), y)
        y = torch.where(((minQ+ref[30]*delta)<=y) & (y<(minQ+ref[31]*delta)), torch.add(base,minQ+quant[31]*delta), y)
        
        
    else:
        y = x.clone()
    return y


def LinearQuantizeOut(x, bit):
    minQ = torch.min(x)
    delta = torch.max(x) - torch.min(x)
    y = x.clone()

    stepSizeRatio = 2.**(-bit)
    stepSize = stepSizeRatio*delta.item()
    index = torch.clamp(torch.floor((x-minQ.item())/stepSize), 0, (2.**(bit)-1))
    y = index*stepSize + minQ.item()

    return y


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

