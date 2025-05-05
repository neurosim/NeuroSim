#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Some helper functions for implementing quantized modules"""
import copy
import inspect

from absl import logging

import torch
from torch import nn
import math
import numpy as np
import csv
from pytorch_quantization.nn import TensorQuantizer
from pytorch_quantization.tensor_quant import QuantDescriptor, QUANT_DESC_8BIT_PER_TENSOR

class QuantMixin():
    """Mixin class for adding basic quantization logic and cim parameters to quantized modules"""

    default_quant_desc_input      = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight     = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_adc        = QUANT_DESC_8BIT_PER_TENSOR

    @classmethod
    def set_default_quant_desc_input(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_input = copy.deepcopy(value)

    @classmethod
    def set_default_quant_desc_weight(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_weight = copy.deepcopy(value)

    @classmethod
    def set_default_quant_desc_adc(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_adc = copy.deepcopy(value)    

    @classmethod
    def set_default_cim_args(cls, value):
        """
        Args:
            value: An instance of :class:`CIMArgs <pytorch_quantization.cim.modules.args.CIMArgs>`
        """
        cls.default_cim_args = copy.deepcopy(value)  

    def init_cim(self, cim_args, in_features, out_features):
        if not inspect.stack()[1].function == "__init__":
            raise TypeError("{} should be only called by __init__ of quantized module.".format(__name__))
        
        # TODO: Update CIMArgs to be a new class called CIM that where we construct cim_args from a cim_descriptor
        self._cim_args = copy.deepcopy(cim_args)
        
        # check the mean and std conductance
        if cim_args.mem_states_file: # external mean and std
            with open(cim_args.mem_states_file, 'r') as file:
                reader = csv.reader(file)
                header = next(reader)  # Read the header row
                # Check if required columns exist
                if "mean" in header:
                    mean_index = header.index("mean")
                    if "std" in header:
                        std_index = header.index("std")
                    
                    # Initialize lists to store values
                    stds = []
                    means = []

                    # Read rows and extract the required columns
                    for row in reader:
                        stds.append(float(row[std_index]))
                        means.append(float(row[mean_index]))

                    if (len(means)!=2**(cim_args.bitcell)):
                        print("The states in the external file do not match the bitcell.")
                        exit(-1)

                    self.mem_mean_gaussian = torch.tensor(means, dtype=torch.float32, device='cuda')
                    self.mem_std_gaussian = torch.tensor(stds, dtype=torch.float32, device='cuda')
                    self._cim_args.mem_states = torch.tensor(means, dtype=torch.float32, device='cuda')
                    if (self.mem_std_gaussian.sum() != 0):
                        self.mem_sampling_seed = torch.randint(1, 1000, (self._cim_args.weight_precision,))
                    else:
                        self.mem_sampling_seed = torch.zeros(0)
                else:
                    print("The required columns 'mean' is not in the CSV file.")
                    exit(-1)

        # Set intermediate states based on on and off state
        else:
            self.mem_sampling_seed = torch.zeros(0)

            num_states = 2**self._cim_args.bitcell
            self._cim_args.mem_step = (self._cim_args.on_state - self._cim_args.off_state) / (num_states-1)
            self._cim_args.mem_states = torch.zeros(num_states, device='cuda')
            self._cim_args.mem_states[0] = self._cim_args.off_state
            for s in range(num_states):
                self._cim_args.mem_states[s] = self._cim_args.off_state + s*self._cim_args.mem_step
        
        # Initialize variables for analog MAC and ADC sensing

        # Calculate feedback resistance in amplifier for voltage-sensing mode
        if self._cim_args.mem_type == 'resistive':
            self._cim_args.amp_feedback = 1/(self._cim_args.parallel_read*(self._cim_args.mem_states[-1]-self._cim_args.mem_states[0]))

        # Calculate feedback capacitor in amplifier for voltage-sensing mode
        if self._cim_args.mem_type == 'capacitive':
            self._cim_args.amp_feedback = self._cim_args.parallel_read*(self._cim_args.mem_states[-1]-self._cim_args.mem_states[0])

        self._cim_args.weight2d_shape = [in_features, out_features]
        if self._cim_args.parallel_read is None:
            self._cim_args.parallel_read = self._cim_args.weight2d_shape[0]

        if self._cim_args.parallel_read > self._cim_args.sub_array[0]:
            print('WARNING: parallel read is larger than sub array size, reducing parallel read to match...')
            self._cim_args.parallel_read = self._cim_args.sub_array[0]

        # Calculate number of arrays needed (doesn't count reference arrays)
        self._cim_args.num_arrays_y = math.ceil(in_features / cim_args.parallel_read)
        self._cim_args.num_arrays_x = math.ceil(out_features*self._cim_args.weight_precision / cim_args.sub_array[1])

        if (cim_args.rate_stuck_0>0 or cim_args.rate_stuck_1>0):
            self.stuck_sampling_seed = torch.randint(1, 1000, (1,))
            print("Stuck at fault sampling initializing...\n")


        if (cim_args.output_noise == -1.0):
            with open(cim_args.output_noise_file, 'r') as file:
                reader = csv.reader(file)
                header = next(reader)  # Read the header row
                # Check if required columns exist
                if "output" in header:
                    value_index = header.index("output")
                    if "std" in header:
                        std_index = header.index("std")
                    if "mean" in header:
                        mean_index = header.index("mean")
                    # Initialize lists to store values
                    values = []
                    stds = []
                    means = []

                    # Read rows and extract the required columns
                    for row in reader:
                        values.append(float(row[value_index]))
                        stds.append(float(row[std_index]))
                        means.append(float(row[mean_index]))

                    if (len(values)>2**(cim_args.adc_precision)+1):
                        print("The required 'output' numbers are larger than the ADC precision.")
                        exit(-1)

                    # Convert lists to PyTorch tensors
                    self.output_values = torch.tensor(values, dtype=torch.float32, device='cuda')
                    if stds:
                        self.output_noise_stds = torch.tensor(stds, dtype=torch.float32, device='cuda')
                    else:
                        self.output_noise_stds = torch.zeros(0)
                    if means:
                        self.output_noise_means = torch.tensor(means, dtype=torch.float32, device='cuda')
                    else:
                        self.output_noise_means = torch.zeros(0)
                else:
                    print("The required columns 'output' is not in the CSV file.")
                    exit(-1)
                print("Output noise initializing...\n")

        # MLC
        self._cim_args.cells_per_weight = math.ceil(self._cim_args.weight_precision / self._cim_args.bitcell)
        self._cim_args.cells_per_weight = math.ceil(self._cim_args.weight_precision / self._cim_args.bitcell)
        self._cim_args.base = 2**self._cim_args.bitcell
        self._cim_args.cycles_per_input = math.ceil(self._cim_args.input_precision / self._cim_args.dac_precision)
        self._cim_args.input_base = 2**self._cim_args.dac_precision
        

    def init_quantizer(self, quant_desc_input, quant_desc_weight, quant_desc_adc, num_layers=None, num_adc_quantizers=None):
        """Helper function for __init__ of quantized module

        Create input and weight quantizer based on quant_desc passed by kwargs, or default of the class.

        Args:
            quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
            quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
            quant_desc_adc: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`            
            num_layers: An integer. Default None. If not None, create a list of quantizers.
        """
        if not inspect.stack()[1].function == "__init__":
            raise TypeError("{} should be only called by __init__ of quantized module.".format(__name__))
        self._fake_quant = True
        if (not quant_desc_input.fake_quant) or (not quant_desc_weight.fake_quant) or (not quant_desc_adc.fake_quant):
            raise ValueError("Only fake quantization is supported!")

        logging.info("Input is %squantized to %d bits in %s with axis %s!", ""
                     if not quant_desc_input.fake_quant else "fake ",
                     quant_desc_input.num_bits, self.__class__.__name__, quant_desc_input.axis)
        logging.info("Weight is %squantized to %d bits in %s with axis %s!", ""
                     if not quant_desc_weight.fake_quant else "fake ",
                     quant_desc_weight.num_bits, self.__class__.__name__, quant_desc_weight.axis)
        logging.info("Output is %squantized to %d bits in %s with axis %s!", ""
                     if not quant_desc_adc.fake_quant else "fake ",
                     quant_desc_adc.num_bits, self.__class__.__name__, quant_desc_adc.axis)

        if num_layers is None:
            self._input_quantizer = TensorQuantizer(quant_desc_input)
            self._weight_quantizer = TensorQuantizer(quant_desc_weight)
        else:
            self._input_quantizers = nn.ModuleList([TensorQuantizer(quant_desc_input) for _ in range(num_layers)])
            self._weight_quantizers = nn.ModuleList([TensorQuantizer(quant_desc_weight) for _ in range(num_layers)])

        if num_adc_quantizers is None:
            self._adc_quantizer = TensorQuantizer(quant_desc_adc)
        else:
            self._adc_quantizers = nn.ModuleList([TensorQuantizer(quant_desc_adc) for _ in range(num_adc_quantizers)])

    # pylint:disable=missing-docstring
    @property
    def input_quantizer(self):
        return self._input_quantizer

    @property
    def weight_quantizer(self):
        return self._weight_quantizer
    # pylint:enable=missing-docstring

    @property
    def adc_quantizer(self):
        return self._adc_quantizer


class QuantInputMixin():
    """Mixin class for adding basic quantization logic to quantized modules"""

    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR

    @classmethod
    def set_default_quant_desc_input(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_input = copy.deepcopy(value)

    @classmethod
    def set_default_cim_args(cls, value):
        """
        Args:
            value: An instance of :class:`CIMArgs <pytorch_quantization.cim.modules.args.CIMArgs>`
        """
        cls.default_cim_args = copy.deepcopy(value) 

    def init_cim(self, cim_args):
        if not inspect.stack()[1].function == "__init__":
            raise TypeError("{} should be only called by __init__ of quantized module.".format(__name__))
        
        # TODO: Update CIMArgs to be a new class called CIM that where we construct cim_args from a cim_descriptor
        self._cim_args = copy.deepcopy(cim_args)

    def init_quantizer(self, quant_desc_input):
        """Helper function for __init__ of simple quantized module

        Create input quantizer based on quant_desc passed by kwargs, or default of the class.

        Args:
            quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not inspect.stack()[1].function == "__init__":
            raise TypeError("{} should be only called by __init__ of quantized module.".format(__name__))
        self._fake_quant = True
        if not quant_desc_input.fake_quant:
            raise ValueError("Only fake quantization is supported!")

        logging.info("Input is %squantized to %d bits in %s with axis %s!", ""
                     if not quant_desc_input.fake_quant else "fake ",
                     quant_desc_input.num_bits, self.__class__.__name__, quant_desc_input.axis)

        self._input_quantizer = TensorQuantizer(quant_desc_input)
        self._adc_quantizer = None

    # pylint:disable=missing-docstring
    @property
    def input_quantizer(self):
        return self._input_quantizer
    # pylint:enable=missing-docstring


def pop_quant_desc_in_kwargs(quant_cls, input_only=False, **kwargs):
    """Pop quant descriptors in kwargs

    If there is no descriptor in kwargs, the default one in quant_cls will be used

    Arguments:
       quant_cls: A class that has default quantization descriptors
       input_only: A boolean. If True, pop quant_desc_input only, not quant_desc_weight. Default false.

    Keyword Arguments:
       quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
           Quantization descriptor of input.
       quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
           Quantization descriptor of weight.
    """
    quant_desc_input = kwargs.pop('quant_desc_input', quant_cls.default_quant_desc_input)
    if not input_only:
        quant_desc_weight = kwargs.pop('quant_desc_weight', quant_cls.default_quant_desc_weight)
        quant_desc_adc = kwargs.pop('quant_desc_adc', quant_cls.default_quant_desc_adc)

    cim_args = kwargs.pop('cim_args', quant_cls.default_cim_args)

    # Check if anything is left in **kwargs
    if kwargs:
        raise TypeError("Unused keys: {}".format(kwargs.keys()))

    if input_only:
        return quant_desc_input, cim_args
    return quant_desc_input, quant_desc_weight, quant_desc_adc, cim_args
