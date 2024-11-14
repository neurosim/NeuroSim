import torch
import torch.nn as nn
import torch.nn.functional as F
from utee import wage_initializer,wage_quantizer,float_quantizer
import numpy as np

class FConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, logger=None,
                wl_input =8,wl_weight= 8,inference=0,onoffratio=10,cellBit=1,subArray=128,ADCprecision=5,
                vari=0,t=0,v=0,detect=0,target=0,debug = 0, cuda=True, name = 'Fconv'):
        super(FConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.logger = logger
        self.wl_weight = wl_weight
        self.wl_input = wl_input
        self.inference = inference
        self.onoffratio = onoffratio
        self.cellBit = cellBit
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.cuda = cuda
        self.name = name
        
    def forward(self, input):  
        if self.inference == 1:
            weight = float_quantizer.float_range_quantize(self.weight,self.wl_weight)
            weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            input = float_quantizer.float_range_quantize(input,self.wl_input)
            output= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
        else:
            output= F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)        

        return output


class FLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False,logger = None,
	             wl_input =8,wl_weight= 8,inference=0,onoffratio=10,cellBit=1,subArray=128,ADCprecision=5,
                 vari=0,t=0,v=0,detect=0,target=0, cuda=True, name ='Flinear' ):
        super(FLinear, self).__init__(in_features, out_features, bias)
        self.logger = logger
        self.wl_weight = wl_weight
        self.wl_input = wl_input
        self.inference = inference
        self.onoffratio = onoffratio
        self.cellBit = cellBit
        self.subArray = subArray
        self.ADCprecision = ADCprecision
        self.vari = vari
        self.t = t
        self.v = v
        self.detect = detect
        self.target = target
        self.cuda = cuda
        self.name = name

    def forward(self, input):
        if self.inference == 1:
            weight = float_quantizer.float_range_quantize(self.weight,self.wl_weight)
            weight = wage_quantizer.Retention(weight,self.t,self.v,self.detect,self.target)
            input = float_quantizer.float_range_quantize(input,self.wl_input)
            output= F.linear(input, self.weight, self.bias)
            output = wage_quantizer.LinearQuantizeOut(output, self.ADCprecision)
        else:
            output= F.linear(input, self.weight, self.bias)
        return output

