#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""Quantized Pooling
Base code is from nn.pooling, details of Module and original argument can be found there.
Module names are intentionally kept same as unquantized version so that they can be dropped into preexisting model
easily, and load pretrained weight. Aliases with Quant prefix are defined and are encouraged to be used explicitly
when start scratch.
"""

from torch.nn.modules import pooling
import pytorch_quantization.cim.modules.macro as macro
import pytorch_quantization.cim.modules._utils as _cim_utils
# import pytorch_quantization.cim.modules.args as args

from . import _utils

__all__ = [
    "MaxPool1d", "CIMMaxPool1d", "MaxPool2d", "CIMMaxPool2d", "MaxPool3d", "CIMMaxPool3d",
    "AvgPool1d", "CIMAvgPool1d", "AvgPool2d", "CIMAvgPool2d", "AvgPool3d", "CIMAvgPool3d",
    "AdaptiveAvgPool1d", "CIMAdaptiveAvgPool1d", "AdaptiveAvgPool2d", "CIMAdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "CIMAdaptiveAvgPool3d"    
]

def add_pool(model_name):
    filename = './NeuroSIM/NetWork_'+str(model_name)+'.csv'
    with open(filename, 'r') as f:
        lines = f.readlines()

    # most recent layer is followed by a pooling layer
    lines[-1] = lines[-1].replace(',0,', ',1,')
    
    with open(filename, 'w') as f:
        f.writelines(lines)

# TODO: support more pooling modes
class CIMMaxPool1d(pooling.MaxPool1d, macro.CIM, _cim_utils.QuantInputMixin):
    """Quantized 1D maxpool"""

    # default_cim_args = args.CIMArgs() # TODO: update this

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, **kwargs):
        super(CIMMaxPool1d, self).__init__(kernel_size, stride, padding, dilation,
                                             return_indices, ceil_mode)
        
        quant_desc_input, cim_args = _cim_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        
        self.init_quantizer(quant_desc_input)
        self.init_cim(cim_args) 

    def forward(self, input):
        quant_input = self._input_quantizer(input)

        if self._cim_args.write_network:
            # write to NetWork.csv
            add_pool(self._cim_args.model)
            self._cim_args.write_network = False

        return super(CIMMaxPool1d, self).forward(quant_input)
    
class CIMMaxPool2d(pooling.MaxPool2d, macro.CIM, _cim_utils.QuantInputMixin):
    """Quantized 2D maxpool"""

    # default_cim_args = args.CIMArgs() # TODO: update this

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, **kwargs):
        super(CIMMaxPool2d, self).__init__(kernel_size, stride, padding, dilation,
                                             return_indices, ceil_mode)
        
        quant_desc_input, cim_args = _cim_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        
        self.init_quantizer(quant_desc_input)
        self.init_cim(cim_args) 

    def forward(self, input):
        quant_input = self._input_quantizer(input)

        if self._cim_args.write_network:
            # write to NetWork.csv
            add_pool(self._cim_args.model)
            self._cim_args.write_network = False

        return super(CIMMaxPool2d, self).forward(quant_input)

class CIMMaxPool3d(pooling.MaxPool3d, macro.CIM, _cim_utils.QuantInputMixin):
    """Quantized 3D maxpool"""

    # default_cim_args = args.CIMArgs() # TODO: update this

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, **kwargs):
        super(CIMMaxPool3d, self).__init__(kernel_size, stride, padding, dilation,
                                             return_indices, ceil_mode)
        
        quant_desc_input, cim_args = _cim_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        
        self.init_quantizer(quant_desc_input)
        self.init_cim(cim_args) 

    def forward(self, input):
        quant_input = self._input_quantizer(input)

        if self._cim_args.write_network:
            # write to NetWork.csv
            add_pool(self._cim_args.model)
            self._cim_args.write_network = False

        return super(CIMMaxPool3d, self).forward(quant_input)
    
class CIMAvgPool1d(pooling.AvgPool1d, macro.CIM, _cim_utils.QuantInputMixin):
    """Quantized 1D average pool"""
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, **kwargs):
        super(CIMAvgPool1d, self).__init__(kernel_size, stride, padding, ceil_mode,
                                             count_include_pad)
        quant_desc_input, cim_args = _cim_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        
        self.init_quantizer(quant_desc_input)
        self.init_cim(cim_args) 

    def forward(self, input):
        quant_input = self._input_quantizer(input)

        if self._cim_args.write_network:
            # write to NetWork.csv
            add_pool(self._cim_args.model)
            self._cim_args.write_network = False

        return super(CIMAvgPool1d, self).forward(quant_input)
    
class CIMAvgPool2d(pooling.AvgPool2d, macro.CIM, _cim_utils.QuantInputMixin):
    """Quantized 2D average pool"""

    # default_cim_args = args.CIMArgs() # TODO: update this
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None, **kwargs):
        super(CIMAvgPool2d, self).__init__(kernel_size, stride, padding, ceil_mode,
                                             count_include_pad, divisor_override)
        
        quant_desc_input, cim_args = _cim_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        
        self.init_quantizer(quant_desc_input)
        self.init_cim(cim_args) 

    def forward(self, input):
        quant_input = self._input_quantizer(input)

        # TODO: If previous layer is not a linear or conv layer this will mess up the network.csv file

        if self._cim_args.write_network:
            # write to NetWork.csv
            add_pool(self._cim_args.model)
            self._cim_args.write_network = False

        return super(CIMAvgPool2d, self).forward(quant_input)

class CIMAvgPool3d(pooling.AvgPool3d, macro.CIM, _cim_utils.QuantInputMixin):
    """Quantized 3D average pool"""

    # default_cim_args = args.CIMArgs() # TODO: update this
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None, **kwargs):
        super(CIMAvgPool3d, self).__init__(kernel_size, stride, padding, ceil_mode,
                                             count_include_pad, divisor_override)
        
        quant_desc_input, cim_args = _cim_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)
        self.init_cim(cim_args) 

    def forward(self, input):
        quant_input = self._input_quantizer(input)

        # TODO: If previous layer is not a linear or conv layer this will mess up the network.csv file

        if self._cim_args.write_network:
            # write to NetWork.csv
            add_pool(self._cim_args.model)
            self._cim_args.write_network = False

        return super(CIMAvgPool3d, self).forward(quant_input)
    
class CIMAdaptiveAvgPool1d(pooling.AdaptiveAvgPool1d, macro.CIM, _utils.QuantInputMixin):
    """Quantized 1D adaptive average pool"""
    def __init__(self, output_size, **kwargs):
        super(CIMAdaptiveAvgPool1d, self).__init__(output_size)

        quant_desc_input, cim_args = _cim_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)
        self.init_cim(cim_args) 

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        if self._cim_args.write_network:
            # write to NetWork.csv
            add_pool(self._cim_args.model)
            self._cim_args.write_network = False
        return super(CIMAdaptiveAvgPool1d, self).forward(quant_input)

class CIMAdaptiveAvgPool2d(pooling.AdaptiveAvgPool2d, macro.CIM, _utils.QuantInputMixin):
    """Quantized 2D adaptive average pool"""
    def __init__(self, output_size, **kwargs):
        super(CIMAdaptiveAvgPool2d, self).__init__(output_size)

        quant_desc_input, cim_args = _cim_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)
        self.init_cim(cim_args) 

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        if self._cim_args.write_network:
            # write to NetWork.csv
            add_pool(self._cim_args.model)
            self._cim_args.write_network = False
        return super(CIMAdaptiveAvgPool2d, self).forward(quant_input)
    
class CIMAdaptiveAvgPool3d(pooling.AdaptiveAvgPool3d, macro.CIM, _utils.QuantInputMixin):
    """Quantized 3D adaptive average pool"""
    def __init__(self, output_size, **kwargs):
        super(CIMAdaptiveAvgPool3d, self).__init__(output_size)

        quant_desc_input, cim_args = _cim_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)
        self.init_cim(cim_args) 

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        if self._cim_args.write_network:
            # write to NetWork.csv
            add_pool(self._cim_args.model)
            self._cim_args.write_network = False
        return super(CIMAdaptiveAvgPool3d, self).forward(quant_input)
    
AvgPool1d = CIMAvgPool1d
AvgPool2d = CIMAvgPool2d
AvgPool3d = CIMAvgPool3d

MaxPool1d = CIMMaxPool1d
MaxPool2d = CIMMaxPool2d
MaxPool3d = CIMMaxPool3d

AdaptiveAvgPool1d = CIMAdaptiveAvgPool1d
AdaptiveAvgPool2d = CIMAdaptiveAvgPool2d
AdaptiveAvgPool3d = CIMAdaptiveAvgPool3d
