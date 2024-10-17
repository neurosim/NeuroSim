# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:31:35 2020

@author: yando
"""

import numpy as np
from input import get_input
from hardware import layer_performance
from parameters import param

def main(tech_node, buffer_type, buffer_size, google_tpu, input_file):

    buffer_bank_size = 256.0 / 1024   # unit: MB, 256KB by default
    buffer_subarray_size = 128 * 512.0 / 1024 / 1024 / 8 # unit: MB, 256x512 subarray by default
    dram_type = "LPDDR4" # the DRAM interface

    dict_bn_param_size = {"ResNet18": 5800,
                        "VGG8": 0}
    bn_param_bit = 32

    if input_file.find("ResNet20") > -1:
        bn_param_size = dict_bn_param_size["ResNet20"]
    elif input_file.find("ResNet18") > -1:
        bn_param_size = dict_bn_param_size["ResNet18"]
    elif input_file.find("ResNet34") > -1:
        bn_param_size = dict_bn_param_size["ResNet34"]
    elif input_file.find("DenseNet40") > -1:
        bn_param_size = dict_bn_param_size["DenseNet40"]
    elif input_file.find("VGG8") > -1:
        bn_param_size = dict_bn_param_size["VGG8"]

    var_dict_list = get_input(input_file)
    layer_stats = []

    hardware_param = param(buffer_type, buffer_size, tech_node, buffer_bank_size, dram_type, google_tpu)

    bn_param_size *= bn_param_bit # unit: bit    
    bn_param_size *= 4 # 4 stats: mu, sigma, gama, beta
    bn_num_bank = np.ceil(bn_param_size / hardware_param.GB_bank_size)

    for var_dict in var_dict_list:
        # print(var_dict['Layer'])
        layer_stats.append(layer_performance(var_dict, buffer_type, buffer_size, hardware_param, False))
        
    # layer_stats.append(layer_performance(var_dict_list[19], buffer_type, 2))

    total_num_ops = 0
    total_computing_latency = 0
    total_energy = 0
    total_mac_energy = 0
    total_RF_energy = 0
    total_GB_energy = 0
    total_GB_read_energy = 0
    total_GB_write_energy = 0
    total_GB_refresh_energy = 0
    total_NOC_energy = 0
    total_DRAM_energy = 0
    total_standby_energy = 0
    total_fmodule_energy = 0
    mac_utilization = 0

    weight_size = 0
    weight_nv_size = 0
    weight_v_size = 0
    weight_v_non_refresh_size = 0
    weight_nv_layer = 0
    weight_v_layer = 0
    weight_v_non_refresh_layer = 0

    input_size = 0
    input_nv_size = 0
    input_v_size = 0
    input_v_non_refresh_size = 0
    input_nv_layer = 0
    input_v_layer = 0
    input_v_non_refresh_layer = 0

    output_size = 0
    output_nv_size = 0
    output_v_size = 0
    output_v_non_refresh_size = 0
    output_nv_layer = 0
    output_v_layer = 0
    output_v_non_refresh_layer = 0


    for l_stats in layer_stats:
        # print(l_stats.energy_efficiency, "TOPS/W")
        total_num_ops += l_stats.num_ops
        total_computing_latency += l_stats.computing_latency
        total_energy += l_stats.total_energy
        total_mac_energy += l_stats.mac_energy
        total_RF_energy += l_stats.RF_energy
        total_GB_energy += l_stats.GB_energy
        total_GB_read_energy += l_stats.GB_read_energy
        total_GB_write_energy += l_stats.GB_write_energy
        total_GB_refresh_energy += l_stats.GB_refresh_energy
        total_NOC_energy += l_stats.noc_energy
        total_DRAM_energy += l_stats.DRAM_energy
        total_standby_energy += l_stats.standby_energy
        total_fmodule_energy += l_stats.fmodule_energy
        mac_utilization += float(l_stats.num_mac_used) / l_stats.num_mac
        
        # get the portion of data:
        if buffer_type == "nvDRAM":
            weight_size += l_stats.weight_size
            input_size += l_stats.input_size
            output_size += l_stats.output_size
            
            if l_stats.weight_buffer_mode == "nv_mode":
                weight_nv_size += l_stats.weight_size
                weight_nv_layer += 1
            elif l_stats.weight_buffer_mode == "v_mode" and l_stats.weight_buffer_refresh == True:
                weight_v_size += l_stats.weight_size
                weight_v_layer += 1
            elif l_stats.weight_buffer_mode == "v_mode" and l_stats.weight_buffer_refresh == False:
                weight_v_non_refresh_size += l_stats.weight_size       
                weight_v_non_refresh_layer += 1
                
            if l_stats.input_buffer_mode == "nv_mode":
                input_nv_size += l_stats.input_size
                input_nv_layer += 1
            elif l_stats.input_buffer_mode == "v_mode" and l_stats.input_buffer_refresh == True:
                input_v_size += l_stats.input_size
                input_v_layer += 1
            elif l_stats.input_buffer_mode == "v_mode" and l_stats.input_buffer_refresh == False:
                input_v_non_refresh_size += l_stats.input_size         
                input_v_non_refresh_layer += 1         

            if l_stats.output_buffer_mode == "nv_mode":
                output_nv_size += l_stats.output_size
                output_nv_layer += 1
            elif l_stats.output_buffer_mode == "v_mode" and l_stats.output_buffer_refresh == True:
                output_v_size += l_stats.output_size
                output_v_layer += 1
            elif l_stats.output_buffer_mode == "v_mode" and l_stats.output_buffer_refresh == False:
                output_v_non_refresh_size += l_stats.output_size 
                output_v_non_refresh_layer += 1
                
    mac_utilization /= len(layer_stats)
    mac_utilization *= 100
        
    area = layer_stats[0].area
    area_mac = layer_stats[0].mac_area
    area_RF = layer_stats[0].RF_area
    area_GB = layer_stats[0].GB_area
    area_fmodule = layer_stats[0].fmodule_area
    standby_power = layer_stats[0].standby_power
    refresh_power = layer_stats[0].refresh_power
    computing_sparsity = layer_stats[0].param.computing_sparsity 
    energy_efficiency = 2 * total_num_ops / total_energy / 1e12
    area_efficiency = 2 * total_num_ops / total_computing_latency / (area*1e6) / 1e12

    bn_refresh_power = 0
    if buffer_type == "eDRAM" :
        bn_refresh_power = bn_num_bank * hardware_param.GB_bank_size\
                        * hardware_param.GB_refresh_energy_per_bit / hardware_param.GB_refresh_period
    elif buffer_type == "nvDRAM" :
        bn_refresh_power = bn_num_bank * hardware_param.GB_bank_size\
                        * (hardware_param.GB_read_energy_per_bit_nv + hardware_param.GB_write_energy_per_bit_nv
                            + hardware_param.GB_read_energy_per_bit_v + hardware_param.GB_write_energy_per_bit_v)\
                        /1.0 # assume 1s idle
    elif buffer_type == "FRAM":
        bn_refresh_power = bn_num_bank * hardware_param.GB_bank_size\
                        * (hardware_param.GB_read_energy_per_bit + hardware_param.GB_write_energy_per_bit)\
                        /1.0 # assume 1s idle  
    standby_power += bn_refresh_power
    print("----------------------TPU GB benchmarking--------------------")          
    print("Chip Design Spec.")
    print("Clock frequency: %d MHz" % (hardware_param.clock_frequency / 1e6))
    print("PE (16x16 digital)")
    print("---MAC energy: %.2f pJ/op" % (hardware_param.mac_energy_per_op * 1e12))
    print("---RF energy: %.2f fJ/bit"  % (hardware_param.RF_read_energy_per_bit * 1e15))
    print("Global Buffer")
    print("---Buffer type:  %s"      % buffer_type)
    print("---Buffer size:  %.4f MB" % buffer_size)
    print("------Bank size: %.4f MB" % buffer_bank_size)

    if buffer_type != "nvDRAM":
        print("---Read Energy:  %.1f fJ/bit"  % (hardware_param.GB_read_energy_per_bit*1e15))
        print("---Write Energy: %.1f fJ/bit"  % (hardware_param.GB_write_energy_per_bit*1e15))
    else:
        print("---Read Energy(v):    %.1f fJ/bit"  % (hardware_param.GB_read_energy_per_bit_v*1e15))
        print("---Write Energy(v):   %.1f fJ/bit"  % (hardware_param.GB_write_energy_per_bit_v*1e15))
        print("---Read Energy(nv):   %.1f fJ/bit"  % (hardware_param.GB_read_energy_per_bit_nv*1e15))
        print("---Write Energy(nv):  %.1f fJ/bit"  % (hardware_param.GB_write_energy_per_bit_nv*1e15))
    if buffer_type == "eDRAM" or buffer_type == "nvDRAM":
        print("---Refresh Energy(v): %.1f fJ/bit" % (hardware_param.GB_refresh_energy_per_bit*1e15))
        
    print("---Leakage power: %.2f uW/chip" % (hardware_param.GB_standby_power*1e6))


    print("")         
    print("Chip Summary")
    print("Energy efficiency: %.4f TOPS/W" % energy_efficiency)
    print("Area efficiency: %.4f TOPS/mm^2" % area_efficiency)
    print("Computing latency: %.4f us" % (total_computing_latency*1e6))
    print("MAC utilization: %.2f %%" % mac_utilization)
    print("Chip area: %.4f mm^2" % (area*1e6))
    print("---MAC: %.4f mm^2" % (area_mac*1e6))
    print("---local RF: %.4f mm^2" % (area_RF*1e6))
    print("---GB: %.4f mm^2" % (area_GB*1e6))
    print("---Functional module: %.4f mm^2" % (area_fmodule*1e6))
    print("Total energy: %.4f uJ" % (total_energy*1e6))
    print("--- MAC energy: %.4f uJ" % (total_mac_energy*1e6))
    print("--- RF energy: %.4f uJ" % (total_RF_energy*1e6))
    print("--- GB energy: %.4f uJ" % (total_GB_energy*1e6 + total_NOC_energy*1e6))
    print("------ GB refresh energy: %.4f uJ" % (total_GB_refresh_energy*1e6))
    print("------ GB access energy: %.4f uJ" % ((total_GB_energy + total_NOC_energy - total_GB_refresh_energy)*1e6))
    print("--------- Read energy: %.4f uJ" % (total_GB_read_energy*1e6))
    print("--------- Write energy: %.4f uJ" % (total_GB_write_energy*1e6))
    print("--------- NOC energy: %.4f uJ" % (total_NOC_energy*1e6))
    print("--- Functional module: %.4f uJ" % (total_fmodule_energy*1e6))
    print("--- DRAM energy: %.4f uJ" % (total_DRAM_energy*1e6))
    print("--- Standby energy: %.4f uJ" % (total_standby_energy*1e6))


    standby_power_mac = layer_stats[0].standby_power_mac
    standby_power_RF = layer_stats[0].standby_power_RF
    standby_power_GB = layer_stats[0].param.GB_standby_power
    standby_power_fmodule = layer_stats[0].standby_power_fmodule

    print("Standby power: %.4f uW" % (standby_power*1e6))
    print("---MAC units leakage: %.4f uW" % (standby_power_mac*1e6))
    print("---RF leakage: %.4f uW" % (standby_power_RF*1e6))
    print("---GB leakage: %.4f uW" % (standby_power_GB*1e6))
    print("---Fmodules leakage: %.4f uW" % (standby_power_fmodule*1e6)) # negligible
    print("---GB Refresh: %.4f uW" % (bn_refresh_power*1e6))
    print("------Num banks to refresh: %d" % (bn_num_bank))


    if buffer_type == 'nvDRAM':
        print(" ")
        print("Buffer mode for different datatypes")
        print("---Weight")
        print("-----nv mode:             %.2f %% (%d layer) " % (weight_nv_size / weight_size * 100.0, 
                                                                weight_nv_layer))
        print("-----v mode:              %.2f %% (%d layer) " % (weight_v_size / weight_size * 100.0,
                                                                weight_v_layer))
        print("-----v mode (no refresh): %.2f %% (%d layer) " % (weight_v_non_refresh_size / weight_size * 100.0, 
                                                                weight_v_non_refresh_layer))
        print("---Input")
        print("-----nv mode:             %.2f %% (%d layer) " % (input_nv_size / input_size * 100.0,
                                                                input_nv_layer))
        print("-----v mode:              %.2f %% (%d layer) " % (input_v_size / input_size * 100.0, 
                                                                input_v_layer))
        print("-----v mode (no refresh): %.2f %% (%d layer) " % (input_v_non_refresh_size / input_size * 100.0,
                                                                input_v_non_refresh_layer))
        print("---Output")
        print("-----nv mode:             %.2f %% (%d layer) " % (output_nv_size / output_size * 100.0,
                                                                output_nv_layer))
        print("-----v mode:              %.2f %% (%d layer) " % (output_v_size / output_size * 100.0,
                                                                output_v_layer))
        print("-----v mode (no refresh): %.2f %% (%d layer) " % (output_v_non_refresh_size / output_size * 100.0,
                                                                output_v_non_refresh_layer))


    f=open('TPU_benchmark.csv', 'w')
    firstline = np.array([["energy efficiency (TOPS/W)", "area efficiency (TOPS/mm2)", "total latency (us)", "MAC utilization", "area (mm2)", "area MAC (mm2)", "area RF (mm2)", "area GB (mm2)", "area functional module (mm2)", 
                           "total energy (uJ)", "MAC energy (uJ)", "RF energy (uJ)", "GB refresh energy (uJ)", "GB read energy (uJ)", "GB write energy (uJ)", "GB NOC energy (uJ)",
                           "functional module energy (uJ)", "DRAM energy (uJ)", "standby energy (uJ)"]])
    np.savetxt(f, firstline, delimiter=",",fmt='%s')
    results = np.array([energy_efficiency, area_efficiency, (total_computing_latency*1e6), mac_utilization, (area*1e6), (area_mac*1e6), (area_RF*1e6), (area_GB*1e6), (area_fmodule*1e6),
            (total_energy*1e6), (total_mac_energy*1e6), (total_RF_energy*1e6), (total_GB_refresh_energy*1e6), (total_GB_read_energy*1e6), (total_GB_write_energy*1e6), (total_NOC_energy*1e6),
            (total_fmodule_energy*1e6), (total_DRAM_energy*1e6), (total_standby_energy*1e6)])
    np.savetxt(f, results, newline=',', fmt='%.4f')
    # f.write("\n")
    f.close()


if __name__ == "__main__":
    main()
    

