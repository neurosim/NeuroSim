import torch    
import math
import numpy as np
import os
class CIM():
    def simulate_array(self, input2d, weight2d):

        # Shift inputs and weights so they are positive
        shift_input  = -torch.min(input2d)
        shift_weight = -torch.min(weight2d)
        input2d = input2d + shift_input
        weight2d = weight2d + shift_weight        


        # Add a dummy row to the input matrix
        input2d = torch.cat((input2d, torch.full((1,input2d.shape[1]), fill_value=shift_input, device=input2d.device)), dim=0)

        # Add a dummy column to the weight matrix
        weight2d = torch.cat((weight2d, torch.full((weight2d.shape[0],1), fill_value=shift_weight, device=weight2d.device)), dim=1)

        # stuck-at-fault
        if self._cim_args.rate_stuck_0>0 or self._cim_args.rate_stuck_1>0:
            weight2d = self.stuck_at_fault_sample(weight2d)

        if self._cim_args.hook:
            write_layer(self, input2d, weight2d)
            self._cim_args.hook = False
            if self._input_quantizer._disabled == True:
                return None

        num_zeros_y = self._cim_args.num_arrays_y*self._cim_args.parallel_read - weight2d.shape[0]
        num_zeros_x = self._cim_args.num_arrays_x*self._cim_args.sub_array[1]  - weight2d.shape[1]

        # pad input and weight if necessary
        if num_zeros_y > 0:
            input2d  = torch.cat((input2d, torch.zeros((input2d.shape[0], num_zeros_y), device=input2d.device)), dim=1)
            weight2d = torch.cat((weight2d, torch.zeros((num_zeros_y, weight2d.shape[1]), device=weight2d.device)), dim=0)
        
        # reshape input and weight matrices according to partitions
        # TODO: Reshape with num_zeros_x
        input2d  = input2d.reshape(input2d.shape[0], self._cim_args.num_arrays_y, -1).transpose(0,1)
        weight2d = weight2d.reshape(self._cim_args.num_arrays_y, -1, weight2d.shape[1])
        
        # reference column of minimal conductance
        weight_min = torch.zeros(weight2d.shape[0], weight2d.shape[1], 1)
        weight_min = weight_min.to(torch.int32)

        matmul_out = torch.zeros((input2d.shape[1], weight2d.shape[2]), device=input2d.device)
        Psum = torch.zeros_like(matmul_out)

        # calculate outputs for each bit of the input
        for i in range(self._cim_args.cycles_per_input):
            input_scale = self._cim_args.input_base**i
            input = ((input2d.to(torch.int32)) >> self._cim_args.dac_precision*i) & (self._cim_args.input_base-1)
            
            Psum[:,:] = 0

            for j in range(self._cim_args.cells_per_weight):
                weight_scale = self._cim_args.base**j
                weight = ((weight2d.to(torch.int32)) >> self._cim_args.bitcell*j) & (self._cim_args.base-1)

                if self._cim_args.hardware:                                  
                    out = self.ADC_output(input, weight, weight_min, j)
                else:
                    out = torch.matmul(input.to(torch.float32), weight.to(torch.float32))

                out = self._adc_quant(out)
                

                # add output for each partition of crossbar rows
                out = out.sum(dim=0)

                # scale adc output by the weight bit significance
                out *= weight_scale

                # add partial sums together
                out = out.to(Psum.device)
                Psum += out

            # scale partial sum for input bit significance
            Psum *= input_scale

            # add partition output to total output of the sub array
            matmul_out += Psum        

        out_dummy_row = matmul_out[-1,:-1].unsqueeze(0) # extract dummy row
        out_dummy_col = matmul_out[:-1,-1].unsqueeze(1) # extract dummy column
        shift = matmul_out[-1,-1]                       # extract dummy element

        matmul_out = matmul_out[:-1,:-1] # remove dummy row and column   

        # Remove shifts
        matmul_out = matmul_out - out_dummy_row - out_dummy_col + shift 

        return matmul_out

    def stuck_at_fault_sample(self, weights):
        # convert decimal to binary
        binary_weights = weights.unsqueeze(-1)
        bit_significance = (1 << torch.arange(self._cim_args.weight_precision))
        bit_significance = bit_significance.to(torch.int32).to(binary_weights.device)
        binary_weights = (binary_weights & bit_significance) > 0
        binary_weights = binary_weights.int()

        num_cells = weights.shape[0]*weights.shape[1]*self._cim_args.cells_per_weight
        num_stuck = int(num_cells * (self._cim_args.rate_stuck_0+self._cim_args.rate_stuck_1))
        gen = torch.Generator(self.stuck_sampling_seed.device)
        gen.manual_seed(self.stuck_sampling_seed.item())
        indices_flat = torch.randperm(num_cells,generator=gen)[:num_stuck]
        cells_per_weight = self._cim_args.cells_per_weight
        if (self._cim_args.rate_stuck_0>0):
            indices_flat_zeros = indices_flat[:math.ceil(num_stuck*self._cim_args.rate_stuck_0)]
            mask_flat = torch.ones(num_cells, dtype=torch.int32)
            mask_flat[indices_flat_zeros] = 0
            mask = mask_flat.view(weights.shape[0], weights.shape[1], cells_per_weight)
            mask_expanded = torch.ones((weights.shape[0], weights.shape[1], self._cim_args.weight_precision), dtype=torch.int32)
            for i in range(cells_per_weight):
                if i==0 and self._cim_args.weight_precision % self._cim_args.bitcell!=0:
                    lsbs = self._cim_args.weight_precision % self._cim_args.bitcell
                    mask_expanded[...,0:lsbs] = mask[..., i].unsqueeze(-1).expand(-1, -1, lsbs)
                else:
                    mask_expanded[...,i*self._cim_args.bitcell:(i+1)*self._cim_args.bitcell] = mask[..., i].unsqueeze(-1).expand(-1, -1, self._cim_args.bitcell)
            binary_weights = binary_weights & mask_expanded.to(binary_weights.device)
        if (self._cim_args.rate_stuck_1>0):
            indices_flat_ones = indices_flat[math.ceil(num_stuck*self._cim_args.rate_stuck_0):num_stuck]
            mask_flat = torch.zeros(num_cells, dtype=torch.int32)
            mask_flat[indices_flat_ones] = 1
            mask = mask_flat.view(weights.shape[0], weights.shape[1], cells_per_weight)
            mask_expanded = torch.zeros((weights.shape[0], weights.shape[1], self._cim_args.weight_precision), dtype=torch.int32)
            for i in range(cells_per_weight):
                if i==0 and self._cim_args.weight_precision % self._cim_args.bitcell!=0:
                    lsbs = self._cim_args.weight_precision % self._cim_args.bitcell
                    mask_expanded[...,0:lsbs] = mask[..., i].unsqueeze(-1).expand(-1, -1, lsbs)
                else:
                    mask_expanded[...,i*self._cim_args.bitcell:(i+1)*self._cim_args.bitcell] = mask[..., i].unsqueeze(-1).expand(-1, -1, self._cim_args.bitcell)
            binary_weights = binary_weights | mask_expanded.to(binary_weights.device)

        weights_sampling = torch.matmul(binary_weights.to(torch.float32), bit_significance.to(torch.float32))
        weights_sampling = weights_sampling.to(torch.int32).squeeze(-1)

        return weights_sampling
    

    def conductance_sampling(self, weights, weights_ref, weight_significance):
        # conductance variation
        gen = torch.Generator(device=self._cim_args.mem_states.device)  # seed generator for weight conductance
        gen.manual_seed(self.mem_sampling_seed[weight_significance].item())  # fixed seed for the weight conductance
        gen_ref = torch.Generator(device=self._cim_args.mem_states.device)  # seed generator for reference conductance column
        gen_ref.manual_seed(self.mem_sampling_seed[0].item())  # fixed seed for the reference conductance column
        sampled_memory_states = torch.normal(mean=self.mem_mean_gaussian[weights], std=self.mem_std_gaussian[weights],generator=gen)
        sampled_reference_states = torch.normal(mean=self.mem_mean_gaussian[weights_ref], std=self.mem_std_gaussian[weights_ref],generator=gen_ref)

        return sampled_memory_states, sampled_reference_states
    

    def calc_drift(self, memory_states, memory_states_ref, weights):
        # retention

        lower = torch.min(memory_states).item()
        upper = torch.max(memory_states).item()
        lower_ref = torch.min(memory_states_ref).item()
        upper_ref = torch.max(memory_states_ref).item()
        target = (upper - lower)*self._cim_args.target + lower
        target_ref = (upper_ref - lower_ref)*self._cim_args.target + lower_ref
        if self._cim_args.detect == 1: # need to define the sign of v 
            sign = torch.zeros_like(memory_states)
            sign = torch.sign(torch.add(torch.zeros_like(memory_states),target)-memory_states)
            ratio = self._cim_args.t**(self._cim_args.v*sign)
            sign_ref = torch.zeros_like(memory_states_ref)
            sign_ref = torch.sign(torch.add(torch.zeros_like(memory_states_ref),target_ref)-memory_states_ref)
            ratio_ref = self._cim_args.t**(self._cim_args.v*sign_ref)

        else :  # random generate target for each cell
            sign = torch.randint_like(memory_states, -1, 2)
            ratio = self._cim_args.t**(self._cim_args.v*sign)
            sign_ref = torch.randint_like(memory_states_ref, -1, 2)
            ratio_ref = self._cim_args.t**(self._cim_args.v*sign)
        memory_states = torch.clamp((memory_states*ratio), lower, upper)
        memory_states_ref = torch.clamp((memory_states_ref*ratio_ref), lower_ref, upper_ref)
        
        return memory_states, memory_states_ref       

    def ADC_output(self, inputs, weights, weights_ref, weight_significance):
        # assert inputs are binary
        # assert(torch.all(inputs==0 | inputs==1))
        if (self._cim_args.output_noise != 0.0):
            ADC_out_ref = torch.matmul(inputs.to(torch.float32), weights.to(torch.float32)).to(torch.int32)
            ADC_out_ref = torch.clamp(ADC_out_ref, min=0, max=2**self._cim_args.adc_precision)
            if (self._cim_args.output_noise == -1.0):
                ADC_out = torch.normal(mean=self.output_noise_means[ADC_out_ref], std=self.output_noise_stds[ADC_out_ref])
            elif (self._cim_args.output_noise > 0.0):
                ADC_out = torch.normal(mean=ADC_out_ref.to(torch.float32), std=self._cim_args.output_noise)
            else:
                print("Parameter output_noise should be >= 0.0 or =-1.0.")
                exit(-1)
            return torch.clamp(torch.round(ADC_out), min=0.0, max= float(2**self._cim_args.adc_precision))
        
        # Get memory states
        weights = weights.to(self._cim_args.mem_states.device)
        weights_ref = weights_ref.to(self._cim_args.mem_states.device)
        
        memory_states = self._cim_args.mem_states[weights]
        memory_states_ref = self._cim_args.mem_states[weights_ref]

        # weight conductance sampling considering device variation
        if(self.mem_sampling_seed.numel()!=0):
            memory_states, memory_states_ref = self.conductance_sampling(weights, weights_ref, weight_significance)
        
        # device drifting
        if (self._cim_args.v != 0):
            memory_states, memory_states_ref = self.calc_drift(memory_states, memory_states_ref, weights)
        
        # read noise for weight cell and reference cell
        if self._cim_args.read_noise > 0:
            memory_states = torch.normal(mean=memory_states, std=memory_states*self._cim_args.read_noise)
            memory_states_ref = torch.normal(mean=memory_states_ref, std=memory_states_ref*self._cim_args.read_noise)
        
        # subtract off-states 
        memory_states = memory_states - memory_states_ref
        
        # get analog inputs
        V_in = self._cim_args.vdd*inputs.to(torch.float32)

        # Calculate analog outputs (linear relationship (e.g. I=VG or Q=VC))
        V_in = V_in.to(memory_states.device)
        analog_out = V_in @ memory_states

        # Linear I-V or Q-V conversion using amplifier
        V_out = analog_out * self._cim_args.amp_feedback
        
        # Emulate ADC sensing

        # Calculate scale for ADC emulation 
        #  Generate voltage references using reference array
        #  Assume all memory states are linearly separated
        analog_step = self._cim_args.vdd*(self._cim_args.mem_states[1]-self._cim_args.mem_states[0])
        V_step = analog_step*self._cim_args.amp_feedback
        scale  = 1/V_step
        ADC_out = torch.round(scale*V_out)
        ADC_out = torch.clamp(ADC_out,min=0.0,max=float(2**self._cim_args.adc_precision))
        return ADC_out

    def _adc_quant(self, inputs):
        """Apply ADC quantization on input tensor

        Simply clip off any outputs larger than amax

        Arguments:
            input: in_features to quantize
        Returns:
            A tuple: (quant_in_feature, quant_weight)
        """

        if self._adc_quantizer._disabled:
            return inputs
        
        outputs = inputs

        if self._adc_quantizer._if_calib:
            outputs = self._adc_quantizer(inputs)

        else:
            adc_max = 2**self._adc_quantizer.num_bits

            # Clip outputs to adc_max
            outputs = torch.clamp(outputs, min=0, max=adc_max)

            # Uncomment to print errors from ADC quantization
            # quant_error = torch.sum(torch.abs(outputs - inputs))
            # if quant_error > 0:
            #     self._cim_args.logger(f'ADC quantization error: {quant_error}')

        return outputs

def convert_to_n_ary(dec_matrix, base, bits=8):
    # expand each column in the decimal matrix to an n-ary number
    rows, cols = dec_matrix.shape
    dec_matrix = dec_matrix.flatten().reshape(-1,1).int()

    max_val = 2**bits
    num_digits = math.ceil(math.log(max_val, base))

    n_ary = base**torch.arange(num_digits, device=dec_matrix.device).flip(0)

    out = dec_matrix // n_ary % base

    return out.reshape(rows, num_digits*cols)

def dec2bin(x, num_bits):
    x = x.int()

    output = torch.zeros(x.shape[0], x.shape[1], num_bits, device=x.device, dtype=torch.int)
    
    for i in range(num_bits):
        bit = (x >> (num_bits - 1 - i)) & 1
        output[:,:, i] = bit
    
    output = output.view(x.shape[0], x.shape[1]*num_bits) 
    return output   

def dec2val(x, num_bits, cells_per_weight, bitcell, base):
    x = x.int()

    output = torch.zeros(x.shape[0], x.shape[1], cells_per_weight, device=x.device, dtype=torch.int)
    
    for i in range(cells_per_weight):
        val = (x >> bitcell*i) & (base-1)
        output[:,:, i] = val
    
    output = output.view(x.shape[0], x.shape[1]*cells_per_weight)  

    return output

def write_layer(self, input2d, weight2d):
    # inputs and weights are in integer format
    # check the computation is regular linear layer or position bias
    rows_per_input = input2d.shape[0] // self._cim_args.batch_size if input2d.shape[0]>self._cim_args.batch_size else input2d.shape[0]
    input2d = input2d[0:rows_per_input, :].t()

    input_file_name =  './layer_record_' + self._cim_args.model + '/input_' + str(self._cim_args.name) + '.csv'
    weight_file_name =  './layer_record_' + self._cim_args.model + '/weight_' + str(self._cim_args.name) + '.csv'
    f = open('./layer_record_' + self._cim_args.model + '/trace_command.sh', "a")
    f.write(weight_file_name+' '+input_file_name+' ')
    f.close()

    np.savetxt(input_file_name, input2d.int().cpu(), delimiter=",",fmt='%s')
    np.savetxt(weight_file_name, weight2d.int().cpu(), delimiter=",",fmt='%s')

# debug D2D variation
def save_weight_compare(self, weight_state, weight_significance):
    shape_str = "_".join(map(str, self.weight.shape))
    file_name = f"weights_name_{self.name}_sig_{weight_significance}.pt"
    file_path = os.path.join(os.getcwd(), file_name)
    

    if os.path.exists(file_path):
        saved_weight = torch.load(file_path)
        
        if not torch.equal(saved_weight, weight_state):
            raise ValueError(f"file: {file_name} 's weight is different to weight_state")
    else:
        torch.save(weight_state, file_path)
