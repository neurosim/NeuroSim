import numpy as np
import argparse
import subprocess 
import evaluate

parser = argparse.ArgumentParser(description='TPU global buffer benchmarking')
parser.add_argument('--technode', default='22', type=int, help='3 / 7 / 22nm')
parser.add_argument('--memory', default='SRAM', help='refresh is considered when memory == eDRAM / nvDRAM / FRAM') 
parser.add_argument('--tputype', default='edge', help='edge / cloud')
parser.add_argument('--model', default='ResNet18', help='VGG8 / ResNet18')
args = parser.parse_args()


# call NeuroSIM to simulate 256kB buffer
# please manually set memory option to be consistent with technode and memory
subprocess.call(['./NeuroSIM/main', '22nm_SRAM'])
# memory options can be added or modified in ./NeuroSIM/main.cpp setting fuction
# default memory options used in Nature Reviews paper: 
# edge TPU: 22nm_SRAM, 22nm_eDRAM_IWO_fast, 22nm_FeFET_IWO, 22nm_STTMRAM_LLC, 22nm_SOTMRAM
# cloud TPU: 3nm_SRAM, 3D_eDRAM_IWO_fast, 3D_FeFET_IWO, 3nm_STTMRAM_LLC, 3nm_SOTMRAM


# TPU GB benchmarking
tech_node = args.technode
buffer_type = args.memory
# available models: VGG8 for edge TPU, VGG8 for cloud TPU, ResNet18 for edge TPU; all weight stationary dataflow
if (args.tputype=='edge'):
    google_tpu = False 
    buffer_size = 2.0  
    if (args.model=='VGG8'):
        input_file = "./traces/VGG8_ws.csv"
    elif (args.model=='ResNet18'):
        input_file = "./traces/ResNet18_ws.csv"
    else: raise ValueError("unavailable model and tpu type combination")
elif (args.tputype=='cloud'): 
    google_tpu = True
    buffer_size = 24.0
    if (args.model=='VGG8'):
        input_file = "./traces/VGG8_ws_256x256.csv"
    else: raise ValueError("unavailable model and tpu type combination")
else: raise ValueError("unavailable model and tpu type combination")
evaluate.main(tech_node, buffer_type, buffer_size, google_tpu, input_file)