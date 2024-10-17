# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:05:58 2020

@author: yando
"""
import numpy as np

# the parameters for the modules
class param:
    
    def __init__(self, GB_type, GB_size, tech_node, GB_bank_size=0.25, dram='LPDDR4', google_tpu=False):
        
        self.weight_bit = 8 # weight
        self.input_bit = 8  # input image/activation
        self.output_bit = 8 # the output psum bit
        self.computing_sparsity = 1.0 # ignore the energy for sparse input (but no hardware support added)
        
        self.GB_size = GB_size * 8                         # "GB size" unit: MB, * 8 convert to Mbit
        self.num_bank_2Mb = self.GB_size / 2               # number of 2Mbit bank, each bank is 2Mbit
        self.num_bank = self.GB_size / GB_bank_size
        self.GB_bank_size = GB_bank_size * 1024 * 1024 * 8 # bank size in bit
        self.GB_subarray_size = 256 * 512
        
        self.num_relu = 64
        self.num_mp = 64
        self.num_fpu = 64
        
        # DRAM access energy (tech node independent)
        if dram == 'LPDDR3':
            self.DRAM_energy_per_bit  =  4.0 * 1e-12
        elif dram == 'LPDDR4':
            self.DRAM_energy_per_bit  =  1.2 * 1e-12
        else:
            print("[%s] Please specify correct DRAM type: LPDDR3 or LPDDR4" % __file__)
            exit(-1)
           
        # from cacti report, Figure 12 low swing wiring
        # CACTI 6.0: A Tool to Model Large Caches
        # About 12fF/mm/bit
        self.noc_unit_length_energy_per_bit = 12 * 1e-15 * 1e3 
        
        if tech_node == 32:
            # 32nm TPU
            if google_tpu == False:
                self.clock_frequency     = 200 * 1e6  # Custom systolic array: 200MHz
            else:
                self.clock_frequency     = 175 * 1e6  # TPU: 700MHz, but takes 4 cycles for one MAC
            self.cycle_time = 1. / self.clock_frequency
            
            # 512-bit RF
            self.RF_read_energy_per_bit    = 0.00477 * 1e-12   # unit: J
            self.RF_write_energy_per_bit   = 0.00477 * 1e-12
            self.RF_area_per_unit          = 1338.82 * 1e-12  # unit: m^2
            self.RF_standby_power_per_unit = 0.12854 * 1e-6 # unit: W
            
            # self.multiplier_energy_per_op          = 0.8223 * 1e-12   # unit: J # LSTP
            self.multiplier_energy_per_op          = 0.4657 * 1e-12   # unit: J # HP
            self.multiplier_area_per_unit          = 1418.66 * 1e-12  # unit: m^2
            self.multiplier_standby_power_per_unit = 0.00627 * 1e-6 # unit: W
    
            self.adder_energy_per_op          = 0.0163 * 1e-12  # unit: J # HP            
            self.adder_area_per_unit          = 23.534 * 1e-12  # unit: m^2
            self.adder_standby_power_per_unit = 0.00186 * 1e-6   # unit: W

            # other functional modules on chip
            self.fpu_energy_per_op = 6.8 * 1e-12           # 6.8pJ/bit (45nm)
            self.fpu_area = 6127.3 * 1e-12                 # 1 FPU
            self.fpu_standby_power = 0.16 * 1e-3 * 3.66e-3 # for LSTP (From NeuroSim: HP[90]= 3.861 * 1e-4 mA/um LSTP[90]=1.415 * 1e-6mA/um, scale by 3.66 * 1e-3)
            self.fpu_area *= self.num_fpu
            self.fpu_standby_power *= self.num_fpu
            
            self.mp_energy_per_op = 0.0087*1e-12          # The read dynamic energy to do max pooling for the numMaxPoolUnit inputs
            self.mp_area = 124.207 * 1e-12                # The area for maxPoolUnit
            self.mp_standby_power = 9.0378*1e-5*1e-6
            self.mp_area *= self.num_mp
            self.mp_standby_power *= self.num_mp
            
            self.relu_area = 20.919*1e-12                # The area for 64 relu units
            self.relu_energy_per_op = 0.076326*1e-12     # The energy for one relu operation
            self.relu_standby_power = 0.001928*1e-6
            self.relu_area *= self.num_relu
            self.relu_standby_power *= self.num_relu
                
            
        elif tech_node == 22:
            # 22nm TPU
            # print("[hardware_parameters] Only supports real TPU design at 22nm node")
            if google_tpu == False:
                self.clock_frequency = 200 * 1e6 #200 * 1e6  # customized systolic-array-like design    
            else:
                self.clock_frequency = 175 * 1e6  # TPU: 700MHz, but takes 4 cycles for one MAC
            self.cycle_time              = 1. / self.clock_frequency

            self.RF_read_energy_per_bit  = 0.00255 * 1e-12
            self.RF_write_energy_per_bit = 0.00255 * 1e-12    
            self.RF_area_per_unit        = 632.8 * 1e-12  # 512-bit RF             
            self.RF_standby_power_per_unit = 0.08 * 1e-6
            
            self.multiplier_energy_per_op          = 0.4144 * 1e-12 # unit: J # LSTP
            # self.multiplier_energy_per_op          = 0.2639 * 1e-12 # unit: J # HP
            self.multiplier_area_per_unit          = 670.54 * 1e-12  # unit: m^2
            self.multiplier_standby_power_per_unit = 0.00358 * 1e-6
    
            self.adder_energy_per_op          = 0.00858 * 1e-12 # unit: J
            self.adder_area_per_unit          = 11.1235 * 1e-12 # unit: m^2
            self.adder_standby_power_per_unit = 0.00107 * 1e-6
            
            # for other modules on chip (22nm)
            self.fpu_energy_per_op = 6.8 * 1e-12           # 6.8pJ/bit (45nm)
            self.fpu_area = 2896.1 * 1e-12                 # 64 FPU on-chip
            self.fpu_standby_power = 0.16 * 1e-3 * 4.25e-3 # for LSTP (From NeuroSim: HP[90]= 3.96 * 1e-4 mA/um LSTP[90K]=1.68 * 1e-6mA/um, scale by 3.66 * 1e-3)
            self.fpu_area *= self.num_fpu
            self.fpu_standby_power *= self.num_fpu
            
            self.mp_energy_per_op = 0.00464 * 1e-12      # The read dynamic energy to do max pooling for the numMaxPoolUnit inputs
            self.mp_area = 58.707 * 1e-12                # The area for maxPoolUnit
            self.mp_standby_power = 5.86245 * 1e-5 * 1e-6
            self.mp_area *= self.num_mp
            self.mp_standby_power *= self.num_mp
            
            self.relu_area = 9.8875*1e-12                 # The area for 64 relu units
            self.relu_energy_per_op = 0.04083*1e-12       # The energy for one relu operation
            self.relu_standby_power = 0.00125*1e-6
            self.relu_area *= self.num_relu
            self.relu_standby_power *= self.num_relu

                
        elif tech_node == 7:
            # 7nm TPU
            # print("[hardware_parameters] Only supports real TPU design at 7nm node")
            # both systolic array and TPU, assume 250MHz
            if google_tpu == False:
                self.clock_frequency = 200 * 1e6 #200 * 1e6  # customized systolic-array-like design    
            else:
                self.clock_frequency = 700 * 1e6  # TPU: 700MHz, but takes 4 cycles for one MAC     # TPU: 250MHz, assume clk freq increase when scaled down
            self.cycle_time              = 1. / self.clock_frequency

            self.RF_read_energy_per_bit  = 0.000835 * 1e-12
            self.RF_write_energy_per_bit = 0.000835 * 1e-12   
            self.RF_area_per_unit        = 89.35340 * 1e-12  # 512-bit RF 
            self.RF_standby_power_per_unit = 0.06338 * 1e-6
            
            self.multiplier_energy_per_op          = 0.32762 * 1e-12 # unit: J
            self.multiplier_area_per_unit          = 94.6820 * 1e-12 # unit: m^2
            self.multiplier_standby_power_per_unit = 0.00564 * 1e-6
    
            self.adder_energy_per_op          = 0.00235 * 1e-12 # unit: J
            self.adder_area_per_unit          = 1.57067 * 1e-12  # unit: m^2
            self.adder_standby_power_per_unit = 0.000537 * 1e-6
             
            # for other modules on chip
            self.fpu_energy_per_op = 1.547 * 1e-12         # 6.8pJ/bit (45nm) use CV^2 to scale with PTM LSTP model
            self.fpu_area = 1448.05 * 1e-12                # 64 FPU on-chip: use (Lg=11nm/45nm) * (Pfin=22nm/45nm)
            self.fpu_standby_power = 0.16 * 1e-3 * 4.87e-3 # for LSTP. From NeuroSim: HP[90] = 1.796 * 1e-3 mA/um LSTP[90] = 8.7624 * 1e-6 mA/um, scale by 4.87 * 1e-3
            self.fpu_area *= self.num_fpu
            self.fpu_standby_power *= self.num_fpu
            
            self.mp_energy_per_op = 0.00151 *1e-12        # The read dynamic energy to do max pooling for the numMaxPoolUnit inputs
            self.mp_area = 8.46414 * 1e-12                # The area for maxPoolUnit
            self.mp_standby_power = 4.64 * 1e-5 * 1e-6
            self.mp_area *= self.num_mp
            self.mp_standby_power *= self.num_mp
            
            self.relu_area = 1.396*1e-12                # The area for 64 relu units
            self.relu_energy_per_op = 0.01336 * 1e-15   # The energy for one relu operation
            self.relu_standby_power = 0.0099 * 1e-6
            self.relu_area *= self.num_relu
            self.relu_standby_power *= self.num_relu


        elif tech_node == 3:
            # 3nm TPU
            # print("[hardware_parameters] Only supports real TPU design at 3nm node")
            # both systolic array and TPU, assume 250MHz
            if google_tpu == False:
                self.clock_frequency = 200 * 1e6 #200 * 1e6  # customized systolic-array-like design    
            else:
                self.clock_frequency = 700 * 1e6  # TPU: 700MHz, but takes 4 cycles for one MAC     # TPU: 250MHz, assume clk freq increase when scaled down
            self.cycle_time              = 1. / self.clock_frequency

            self.RF_read_energy_per_bit  = 0.000835 * 1e-12 * 0.75  # 7nm -> 3nm
            self.RF_write_energy_per_bit = 0.000835 * 1e-12 * 0.75   
            self.RF_area_per_unit        = 89.35340 * 1e-12 * 0.36  # 512-bit RF    #7nm -> 3nm feature size 12*12/20/20 = 0.36
            self.RF_standby_power_per_unit = 0.06338 * 1e-6
            
            self.multiplier_energy_per_op          = 0.32762 * 1e-12 * 0.75 # unit: J
            self.multiplier_area_per_unit          = 94.6820 * 1e-12 * 0.36 # unit: m^2
            self.multiplier_standby_power_per_unit = 0.00564 * 1e-6
    
            self.adder_energy_per_op          = 0.00235 * 1e-12 * 0.75 # unit: J
            self.adder_area_per_unit          = 1.57067 * 1e-12 * 0.36  # unit: m^2
            self.adder_standby_power_per_unit = 0.000537 * 1e-6
             
            # for other modules on chip
            self.fpu_energy_per_op = 1.547 * 1e-12 * 0.75         # 6.8pJ/bit (45nm) use CV^2 to scale with PTM LSTP model
            self.fpu_area = 1448.05 * 1e-12 * 0.36                # 64 FPU on-chip: use (Lg=11nm/45nm) * (Pfin=22nm/45nm)
            self.fpu_standby_power = 0.16 * 1e-3 * 4.87e-3 # HP unavailable for 3nm. follow the number of 7nm   # for LSTP. From NeuroSim: HP[90] = 1.796 * 1e-3 mA/um LSTP[90] = 8.7624 * 1e-6 mA/um, scale by 4.87 * 1e-3
            self.fpu_area *= self.num_fpu
            self.fpu_standby_power *= self.num_fpu
            
            self.mp_energy_per_op = 0.00151 *1e-12 * 0.75        # The read dynamic energy to do max pooling for the numMaxPoolUnit inputs
            self.mp_area = 8.46414 * 1e-12 * 0.36                # The area for maxPoolUnit
            self.mp_standby_power = 4.64 * 1e-5 * 1e-6
            self.mp_area *= self.num_mp
            self.mp_standby_power *= self.num_mp
            
            self.relu_area = 1.396*1e-12 * 0.36                # The area for 64 relu units
            self.relu_energy_per_op = 0.01336 * 1e-15 * 0.75   # The energy for one relu operation
            self.relu_standby_power = 0.0099 * 1e-6
            self.relu_area *= self.num_relu
            self.relu_standby_power *= self.num_relu

        file = open('./neurosim.csv', 'r')
        GB_neurosim = np.loadtxt(file, delimiter=",")
        self.GB_type = GB_type
        self.GB_area = GB_neurosim[0] * self.num_bank_2Mb
        self.GB_read_energy_per_bit  = GB_neurosim[1] / 512   
        self.GB_write_energy_per_bit = GB_neurosim[2] / 512
        self.GB_standby_power        = GB_neurosim[3] * self.num_bank_2Mb  
        self.GB_refresh_period         = GB_neurosim[4]
        self.GB_refresh_energy_per_bit = GB_neurosim[5] / 512
        file.close()


        self.mac_energy_per_op          = self.multiplier_energy_per_op + self.adder_energy_per_op
        self.mac_area_per_unit          = self.multiplier_area_per_unit + self.adder_area_per_unit
        self.mac_standby_power_per_unit = self.multiplier_standby_power_per_unit + self.adder_standby_power_per_unit
