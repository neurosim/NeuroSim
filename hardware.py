# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:15:31 2020

@author: yando
"""

import numpy as np

class layer_performance:
    
    def __init__(self, var_dict, buffer_type, buffer_size, param, print_stats=True):
        
        self.buffer_type = buffer_type
        self.buffer_size = buffer_size
        
        self.param = param
        
        # unpack the input parameters
        self.name = var_dict['Layer']
        self.num_mac = var_dict['num. MACs']
        self.num_mac_used = var_dict['num. MACs used']
        self.computing_cycles = var_dict['computing cycles']
        self.total_computing_cycles = var_dict['total computing cycle']
         
        self.num_RF_weight = var_dict['num weight RF'] 
        self.num_RF_read_weight = var_dict['weight num RF read'] # read per RF 
        self.num_RF_write_weight = var_dict['weight num RF update'] + var_dict['weight num RF write']
        
        self.num_RF_input = var_dict['num input RF']
        self.num_RF_read_input = var_dict['input num RF read'] 
        self.num_RF_write_input = var_dict['input num RF update'] + var_dict['input num RF write'] 
     
        self.num_RF_output = var_dict['num output RF'] 
        self.num_RF_read_output = var_dict['output num RF read'] 
        self.num_RF_write_output = var_dict['output num RF update'] + var_dict['output num RF write'] 
    
        self.weight_size = var_dict['weight size (GB)'] 
        self.num_GB_read_weight = var_dict['weight num GB read'] 
        self.num_GB_write_weight = var_dict['weight num GB update'] + var_dict['weight num GB write'] 
    
        self.input_size = var_dict['input size (GB)'] 
        self.num_GB_read_input = var_dict['input num GB read'] 
        self.num_GB_write_input = var_dict['input num GB update'] + var_dict['input num GB write'] 
     
        self.output_size = var_dict['output size (GB)'] 
        self.num_GB_read_output = var_dict['output num GB read'] 
        self.num_GB_write_output = var_dict['output num GB update'] +  var_dict['output num GB write'] 
    
        self.num_DRAM_read_weight = var_dict['weight num DRAM read'] 
        self.num_DRAM_write_weight = var_dict['weight num DRAM update'] + var_dict['weight num DRAM write'] 
    
        self.num_DRAM_read_input = var_dict['input num DRAM read'] 
        self.num_DRAM_write_input = var_dict['input num DRAM update'] + var_dict['input num DRAM write'] 
    
        self.num_DRAM_read_output = var_dict['output num DRAM read'] 
        self.num_DRAM_write_output = var_dict['output num DRAM update'] + var_dict['output num DRAM write'] 
    
        self.RF_read_BW_weight = float(var_dict['weight RF read BW (words/cycle)'].split("words")[0].strip())
        self.RF_write_BW_weight = float(var_dict['weight RF write BW (words/cycle)'].split("words")[0].strip())    
        self.RF_read_BW_input = float(var_dict['input RF read BW (words/cycle)'].split("words")[0].strip())
        self.RF_write_BW_input = float(var_dict['input RF write BW (words/cycle)'].split("words")[0].strip())
        self.RF_read_BW_output = float(var_dict['output RF read BW (words/cycle)'].split("words")[0].strip())
        self.RF_write_BW_output = float(var_dict['output RF write BW (words/cycle)'].split("words")[0].strip())
        
        self.GB_read_BW_weight = float(var_dict['weight GB read BW (words/cycle)'].split("words")[0].strip())
        self.GB_write_BW_weight = float(var_dict['weight GB write BW (words/cycle)'].split("words")[0].strip())    
        self.GB_read_BW_input = float(var_dict['input GB read BW (words/cycle)'].split("words")[0].strip())
        self.GB_write_BW_input = float(var_dict['input GB write BW (words/cycle)'].split("words")[0].strip())
        self.GB_read_BW_output = float(var_dict['output GB read BW (words/cycle)'].split("words")[0].strip())
        self.GB_write_BW_output = float(var_dict['output GB write BW (words/cycle)'].split("words")[0].strip()) 
    
        self.DRAM_read_BW_weight = float(var_dict['weight DRAM read BW (words/cycle)'].split("words")[0].strip())
        self.DRAM_write_BW_weight = float(var_dict['weight DRAM write BW (words/cycle)'].split("words")[0].strip())    
        self.DRAM_read_BW_input = float(var_dict['input DRAM read BW (words/cycle)'].split("words")[0].strip())
        self.DRAM_write_BW_input = float(var_dict['input DRAM write BW (words/cycle)'].split("words")[0].strip())
        self.DRAM_read_BW_output = float(var_dict['output DRAM read BW (words/cycle)'].split("words")[0].strip())
        self.DRAM_write_BW_output = float(var_dict['output DRAM write BW (words/cycle)'].split("words")[0].strip())
        
        self.evaluate_hardware_performance()
        
        if print_stats == True:
            self.print_stats()
        
    def evaluate_hardware_performance(self):
        
        # get the chip area
        self.mac_area = self.num_mac * self.param.mac_area_per_unit
        self.RF_area  = self.num_mac * self.param.RF_area_per_unit
        self.GB_area = self.param.GB_area
        self.fmodule_area = self.param.fpu_area + self.param.mp_area + self.param.relu_area
        # self.area = max(self.mac_area + self.RF_area + self.fmodule_area, self.GB_area)    //BEOL
        self.area = self.mac_area + self.RF_area + self.GB_area + self.fmodule_area
        
        # calculate the noc energy based on the mac unit dimension
        noc_energy_per_bit = self.param.noc_unit_length_energy_per_bit * np.sqrt(self.mac_area + self.RF_area)
        
        # computing latency
        self.computing_latency = self.total_computing_cycles * self.param.cycle_time
        
        # energy for mac
        self.num_ops = self.computing_cycles * self.num_mac_used
        self.mac_energy =  self.num_ops * self.param.mac_energy_per_op * self.param.computing_sparsity
        
        # energy for RF
        self.total_num_RF_read_weight = self.num_RF_weight * self.num_RF_read_weight
        self.total_num_RF_write_weight = self.num_RF_weight * self.num_RF_write_weight
        self.RF_read_energy_weight = self.total_num_RF_read_weight * self.param.weight_bit * self.param.RF_read_energy_per_bit
        self.RF_write_energy_weight = self.total_num_RF_write_weight * self.param.weight_bit * self.param.RF_write_energy_per_bit
        
        self.total_num_RF_read_input = self.num_RF_input * self.num_RF_read_input
        self.total_num_RF_write_input = self.num_RF_input * self.num_RF_write_input
        self.RF_read_energy_input = self.total_num_RF_read_input * self.param.input_bit * self.param.RF_read_energy_per_bit
        self.RF_write_energy_input = self.total_num_RF_write_input * self.param.input_bit * self.param.RF_write_energy_per_bit

        self.total_num_RF_read_output = self.num_RF_output * self.num_RF_read_output
        self.total_num_RF_write_output = self.num_RF_output * self.num_RF_write_output
        self.RF_read_energy_output = self.total_num_RF_read_output * self.param.output_bit * self.param.RF_read_energy_per_bit
        self.RF_write_energy_output = self.total_num_RF_write_output * self.param.output_bit * self.param.RF_write_energy_per_bit
        
        self.RF_read_energy = self.RF_read_energy_weight + self.RF_read_energy_input + self.RF_read_energy_output
        self.RF_write_energy = self.RF_write_energy_weight + self.RF_write_energy_input + self.RF_write_energy_output
        self.RF_energy = self.RF_read_energy + self.RF_write_energy
                
        # energy for GB
        self.GB_refresh_energy = 0
        
        self.GB_read_energy_weight = self.num_GB_read_weight * self.param.weight_bit * self.param.GB_read_energy_per_bit
        self.GB_write_energy_weight = self.num_GB_write_weight * self.param.weight_bit * self.param.GB_write_energy_per_bit

        self.GB_read_energy_input = self.num_GB_read_input * self.param.input_bit * self.param.GB_read_energy_per_bit
        self.GB_write_energy_input = self.num_GB_write_input * self.param.input_bit * self.param.GB_write_energy_per_bit
        
        self.GB_read_energy_output = self.num_GB_read_output * self.param.output_bit * self.param.GB_read_energy_per_bit
        self.GB_write_energy_output = self.num_GB_write_output * self.param.output_bit * self.param.GB_write_energy_per_bit
                    
        self.GB_refresh_energy_weight = 0
        self.GB_refresh_energy_input  = 0
        self.GB_refresh_energy_output = 0
        
        if self.buffer_type == "eDRAM":
            num_refresh = np.floor(self.computing_latency / self.param.GB_refresh_period)
            self.GB_refresh_energy_weight = num_refresh * self.param.GB_refresh_energy_per_bit\
                                          * self.weight_size * self.param.weight_bit
            self.GB_refresh_energy_input = num_refresh * self.param.GB_refresh_energy_per_bit\
                                         * self.input_size * self.param.input_bit
            self.GB_refresh_energy_output = num_refresh * self.param.GB_refresh_energy_per_bit\
                                          * self.output_size * self.param.output_bit


        self.GB_read_energy = self.GB_read_energy_weight + self.GB_read_energy_input + self.GB_read_energy_output
        self.GB_write_energy = self.GB_write_energy_weight + self.GB_write_energy_input + self.GB_write_energy_output

                               
        # energy for NOC:
        self.noc_energy_read_weight = self.num_GB_read_weight * self.param.weight_bit * noc_energy_per_bit
        self.noc_energy_write_weight = self.num_GB_write_weight * self.param.weight_bit * noc_energy_per_bit
        self.noc_energy_weight = self.noc_energy_read_weight + self.noc_energy_write_weight
        
        self.noc_energy_read_input = self.num_GB_read_input * self.param.input_bit * noc_energy_per_bit
        self.noc_energy_write_input = self.num_GB_write_input * self.param.input_bit * noc_energy_per_bit
        self.noc_energy_input = self.noc_energy_read_input + self.noc_energy_write_input

        self.noc_energy_read_output = self.num_GB_read_output * self.param.output_bit * noc_energy_per_bit
        self.noc_energy_write_output = self.num_GB_write_output * self.param.output_bit * noc_energy_per_bit
        self.noc_energy_output = self.noc_energy_read_output + self.noc_energy_write_output

        self.noc_energy =  self.noc_energy_weight + self.noc_energy_input + self.noc_energy_output 
        
        # use the refresh power to calculate the refresh energy
        self.GB_energy = self.GB_read_energy + self.GB_write_energy
                           
        # energy for DRAM
        self.DRAM_read_energy_weight = self.num_DRAM_read_weight * self.param.weight_bit * self.param.DRAM_energy_per_bit
        self.DRAM_write_energy_weight = self.num_DRAM_write_weight * self.param.weight_bit * self.param.DRAM_energy_per_bit

        self.DRAM_read_energy_input = self.num_DRAM_read_input * self.param.input_bit * self.param.DRAM_energy_per_bit
        self.DRAM_write_energy_input = self.num_DRAM_write_input * self.param.input_bit * self.param.DRAM_energy_per_bit

        self.DRAM_read_energy_output = self.num_DRAM_read_output * self.param.output_bit * self.param.DRAM_energy_per_bit
        self.DRAM_write_energy_output = self.num_DRAM_write_output * self.param.output_bit * self.param.DRAM_energy_per_bit
        
        self.DRAM_read_energy = self.DRAM_read_energy_weight + self.DRAM_read_energy_input + self.DRAM_read_energy_output
        self.DRAM_write_energy = self.DRAM_write_energy_weight + self.DRAM_write_energy_input + self.DRAM_write_energy_output
        self.DRAM_energy = self.DRAM_read_energy + self.DRAM_write_energy
        
        # the energy for the functional units
        self.fpu_energy = self.output_size * self.param.fpu_energy_per_op * 4 # *4 for batch norm gamma * (x-u)/sqrt(sigma^2 + epsilon) + beta
        self.mp_energy = self.output_size * self.param.mp_energy_per_op
        self.relu_energy = self.output_size * self.param.relu_energy_per_op
        self.fmodule_energy = self.fpu_energy + self.mp_energy + self.relu_energy
        
        # The bandwidth requirement for each memory level
        self.RF_read_BW = self.param.weight_bit * (self.RF_read_BW_weight + self.RF_read_BW_input + self.RF_read_BW_output)
        self.RF_write_BW = self.param.weight_bit * (self.RF_write_BW_weight + self.RF_write_BW_input + self.RF_write_BW_output)

        self.GB_read_BW = self.param.input_bit * (self.GB_read_BW_weight + self.GB_read_BW_input + self.GB_read_BW_output)
        self.GB_write_BW = self.param.input_bit * (self.GB_write_BW_weight + self.GB_write_BW_input + self.GB_write_BW_output)     

        self.DRAM_read_BW = self.param.output_bit * (self.DRAM_read_BW_weight + self.DRAM_read_BW_input + self.DRAM_read_BW_output)
        self.DRAM_write_BW = self.param.output_bit * (self.DRAM_write_BW_weight + self.DRAM_write_BW_input + self.DRAM_write_BW_output)
                
        self.standby_power_mac = self.num_mac * self.param.mac_standby_power_per_unit
        self.standby_power_RF = self.num_mac * self.param.RF_standby_power_per_unit
        self.standby_power_fmodule = self.param.fpu_standby_power + self.param.mp_standby_power\
                                   + self.param.relu_standby_power
        self.standby_power = self.standby_power_mac + self.standby_power_RF + self.param.GB_standby_power\
                           + self.standby_power_fmodule
        self.refresh_power = 0
        
        if self.buffer_type == "eDRAM" :
            num_active_bank = np.ceil(self.weight_size * self.param.weight_bit / self.param.GB_bank_size)\
                            + np.ceil(self.input_size * self.param.input_bit  / self.param.GB_bank_size)\
                            + np.ceil(self.output_size * self.param.output_bit / self.param.GB_bank_size)

            if num_active_bank > self.param.num_bank:
                num_active_bank = self.param.num_bank
            self.refresh_power = self.param.GB_refresh_energy_per_bit / self.param.GB_refresh_period\
                               * num_active_bank * self.param.GB_bank_size
            self.GB_refresh_energy = self.refresh_power * self.computing_latency

        # calculate the refresh energy of GB with refresh power
        self.GB_energy += self.GB_refresh_energy

        # get the total energy consumption
        self.total_energy = self.mac_energy + self.RF_energy + self.GB_energy\
                          + self.noc_energy + self.DRAM_energy\
                          + self.fmodule_energy
                          
        self.standby_energy = self.standby_power * self.computing_latency
        self.total_energy += self.standby_energy
        
        self.energy_efficiency = 2 * self.num_ops / self.total_energy / 1e12
        self.area_efficiency =   2 * self.num_ops / self.computing_latency / (self.area*1e6) / 1e12
    
    def print_stats(self):
        print(self.name, " stats")
        print("Summary")
        print("---Energy: %.2f uJ" % (self.total_energy*1e6))
        print("---Frequency: %.2f MHz" % (self.param.clock_frequency/1e6))
        print("---Latency: %.2f us (%d cycles)" % (self.computing_latency * 1e6, 
                                                   self.total_computing_cycles))
        print("---Area: %.2f mm^2" % (self.area * 1e6))
        print("---Energy efficiency: %.4f TOPS/W"  % self.energy_efficiency)
        print("---Area efficiency: %.4f TOPS/mm^2" % self.area_efficiency)
        print("Breakdown")        
        print("---MAC unit")
        print("------Num MACs: %d, Num MACs used: %d" % (self.num_mac, self.num_mac_used))
        print("------Computing sparsity: %.2f" % (self.param.computing_sparsity))
        print("------Energy: %.2f uJ" % (self.mac_energy*1e6))
        print("------Latency: %.2f us (%d cycles)" % (self.computing_latency * 1e6,
                                                      self.computing_cycles))
        print("------Area: %.2f mm^2" % (self.mac_area*1e6))
        
        print("---Register file (512-bit)")
        print("------Num RF: %d, Num RF used: %d" %(self.num_mac, self.num_mac_used))
        print("------Energy: %.2f uJ" %(self.RF_energy*1e6))
        print("---------Read Energy: %.2f uJ" % (self.RF_read_energy*1e6))
        print("------------Weight")
        print("---------------Num read: %d" % self.total_num_RF_read_weight)        
        print("---------------Read energy: %.2f uJ" % (self.RF_read_energy_weight*1e6))
        print("------------Input")
        print("---------------Num read: %d" % self.total_num_RF_read_input)        
        print("---------------Read energy: %.2f uJ" % (self.RF_read_energy_input*1e6))        
        print("------------Output")
        print("---------------Num read: %d" % self.total_num_RF_read_output)        
        print("---------------Read energy: %.2f uJ" % (self.RF_read_energy_output*1e6))        
        print("---------Write Energy: %.2f uJ" % (self.RF_write_energy*1e6))
        print("------------Weight")
        print("---------------Num read: %d" % self.total_num_RF_write_weight)        
        print("---------------Read energy: %.2f uJ" % (self.RF_write_energy_weight*1e6))
        print("------------Input")
        print("---------------Num read: %d" % self.total_num_RF_write_input)        
        print("---------------Read energy: %.2f uJ" % (self.RF_write_energy_input*1e6))        
        print("------------Output")
        print("---------------Num read: %d" % self.total_num_RF_write_output)        
        print("---------------Read energy: %.2f uJ" % (self.RF_write_energy_output*1e6))
        print("------Area: %.2f mm^2" %(self.RF_area*1e6))        

        print("---Global buffer")
        print("------Energy: %.2f uJ" %(self.GB_energy*1e6))
        print("---------Read Energy: %.2f uJ" % (self.GB_read_energy*1e6))
        print("------------Weight")
        print("---------------Num read: %d" % self.num_GB_read_weight)        
        print("---------------Read energy: %.2f uJ" % (self.GB_read_energy_weight*1e6))
        print("------------Input")
        print("---------------Num read: %d" % self.num_GB_read_input)        
        print("---------------Read energy: %.2f uJ" % (self.GB_read_energy_input*1e6))        
        print("------------Output")
        print("---------------Num read: %d" % self.num_GB_read_output)        
        print("---------------Read energy: %.2f uJ" % (self.GB_read_energy_output*1e6))        
        print("---------Write Energy: %.2f uJ" % (self.GB_write_energy))
        print("------------Weight")
        print("---------------Num read: %d" % self.num_GB_write_weight)        
        print("---------------Read energy: %.2f uJ" % (self.GB_write_energy_weight*1e6))
        print("------------Input")
        print("---------------Num read: %d" % self.num_GB_write_input)        
        print("---------------Read energy: %.2f uJ" % (self.GB_write_energy_input*1e6))        
        print("------------Output")
        print("---------------Num read: %d" % self.num_GB_write_output)        
        print("---------------Read energy: %.2f uJ" % (self.GB_write_energy_output*1e6))
        print("------Area: %.2f mm^2" %(self.GB_area*1e6))
        
        print("---NOC")
        print("------Energy: %.2f uJ" %(self.noc_energy*1e6))
        print("---------Weight : %.2f uJ"  % (self.noc_energy_weight*1e6))
        print("------------read: %.2f uJ"  % (self.noc_energy_read_weight*1e6))
        print("------------write: %.2f uJ" % (self.noc_energy_write_weight*1e6))
        print("---------Input : %.2f uJ"   % (self.noc_energy_input*1e6))
        print("------------read: %.2f uJ"  % (self.noc_energy_read_input*1e6))
        print("------------write: %.2f uJ" % (self.noc_energy_write_input*1e6))
        print("---------Output : %.2f uJ"  % (self.noc_energy_output*1e6))
        print("------------read: %.2f uJ"  % (self.noc_energy_read_output*1e6))
        print("------------write: %.2f uJ" % (self.noc_energy_write_output*1e6))
        
        print("---Functional modules")
        print("------FPU")
        print("---------Energy: %.2f uJ" % (self.fpu_energy*1e6))
        print("---------Area: %.2f mm^2" % (self.fpu_area*1e6))
        print("------Max pooling")
        print("---------Energy: %.2f uJ" % (self.mp_energy*1e6))
        print("---------Area: %.2f mm^2" % (self.mp_area*1e6))
        print("------ReLu")
        print("---------Energy: %.2f uJ" % (self.relu_energy*1e6))
        print("---------Area: %.2f mm^2" % (self.relu_area*1e6))
        

        print("---DRAM")
        print("------Energy: %.2f uJ" %(self.DRAM_energy*1e6))
        print("---------Read Energy: %.2f uJ" % (self.DRAM_read_energy*1e6))
        print("------------Weight")
        print("---------------Num read: %d" % self.num_DRAM_read_weight)        
        print("---------------Read energy: %.2f uJ" % (self.DRAM_read_energy_weight*1e6))
        print("------------Input")
        print("---------------Num read: %d" % self.num_DRAM_read_input)        
        print("---------------Read energy: %.2f uJ" % (self.DRAM_read_energy_input*1e6))        
        print("------------Output")
        print("---------------Num read: %d" % self.num_DRAM_read_output)        
        print("---------------Read energy: %.2f uJ" % (self.DRAM_read_energy_output*1e6))        
        print("---------Write Energy: %.2f uJ" % (self.DRAM_write_energy))
        print("------------Weight")
        print("---------------Num read: %d" % self.num_DRAM_write_weight)        
        print("---------------Read energy: %.2f uJ" % (self.DRAM_write_energy_weight*1e6))
        print("------------Input")
        print("---------------Num read: %d" % self.num_DRAM_write_input)        
        print("---------------Read energy: %.2f uJ" % (self.DRAM_write_energy_input*1e6))        
        print("------------Output")
        print("---------------Num read: %d" % self.num_DRAM_write_output)        
        print("---------------Read energy: %.2f uJ" % (self.DRAM_write_energy_output*1e6))
        
        print("Bandwidth requrement")        
        print("---Register file")
        print("------Read bandwidth %.2f bit/cycle" % self.RF_read_BW)
        print("------Write bandwidth %.2f bit/cycle" % self.RF_write_BW)
        print("---Global buffer")
        print("------Read bandwidth %.2f bit/cycle" % self.GB_read_BW)
        print("------Write bandwidth %.2f bit/cycle" % self.GB_write_BW)
        print("---DRAM")
        print("------Read bandwidth %.2f bit/cycle" % self.DRAM_read_BW)
        print("------Write bandwidth %.2f bit/cycle" % self.DRAM_write_BW)
        
        print("Standby")
        print("---Power: %.2f mW"  % (self.standby_power*1e3))
        print("---Energy: %.2f uJ" % (self.standby_energy*1e6))