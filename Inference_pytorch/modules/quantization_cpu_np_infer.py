import torch
import torch.nn as nn
import torch.nn.functional as F
from utee import wage_initializer,wage_quantizer
from torch._jit_internal import weak_script_method


class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,logger = None,clip_weight = False,wage_init=False,quantize_weight= False,clip_output =False,quantize_output = False,wl_input =8,wl_activate=8,wl_error=8,wl_weight= 8,debug = 0, name = 'Qconv' ):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.logger = logger
        self.clip_weight = clip_weight
        self.wage_init = wage_init
        self.quantize_weight = quantize_weight
        self.clip_output = clip_output
        self.debug = debug
        self.wl_weight = wl_weight
        self.quantize_output = quantize_output
        self.wl_activate = wl_activate
        self.wl_error = wl_error
        self.wl_input = wl_input
        self.name = name
        self.scale  = wage_initializer.wage_init_(self.weight, self.wl_weight, factor=1.0)

    @weak_script_method
    def forward(self, input):
        weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
        weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()
        output= F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        output = output/self.scale

        output = wage_quantizer.WAGEQuantizer_f(output, self.wl_activate, self.wl_error)
        return output


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False,logger = None,clip_weight = False,wage_init=False,quantize_weight= False,clip_output =False,quantize_output = False,wl_input =8,wl_activate=8,wl_error=8,wl_weight= 8,debug = 0, name ='Qlinear' ):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.logger = logger
        self.clip_weight = clip_weight
        self.wage_init = wage_init
        self.quantize_weight = quantize_weight
        self.clip_output = clip_output
        self.debug = debug
        self.wl_weight = wl_weight
        self.quantize_output = quantize_output
        self.wl_activate = wl_activate
        self.wl_input = wl_input
        self.wl_error = wl_error
        self.name = name
        self.scale  = wage_initializer.wage_init_(self.weight, self.wl_weight, factor=1.0)

    @weak_script_method
    def forward(self, input):
        weight1 = self.weight * self.scale + (self.weight - self.weight * self.scale).detach()
        weight = weight1 + (wage_quantizer.Q(weight1,self.wl_weight) -weight1).detach()

        output = F.linear(input, weight, self.bias)
        output = output/self.scale

        output = wage_quantizer.WAGEQuantizer_f(output,self.wl_activate, self.wl_error)
        return output

