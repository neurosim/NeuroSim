In order to run integrated TPU global buffer simulator, please run "main.py"
Please set technode, memory type, tputype, model, and memory cell type.

TPU memory access count tables (traces) are from Timeloop and are provided directly
There are 3 available traces currently: model=VGG8, tputype=edge; model=VGG8, tputype=cloud; model=ResNet18, tputype=edge

NeuroSIM will be called automatically to simulate buffer performance and results will be stored in "neurosim.csv"
memory option needs to be set manually
default memory options used in Nature Reviews paper: 
edge TPU: 22nm_SRAM, 22nm_eDRAM_IWO_fast, 22nm_FeFET_IWO, 22nm_STTMRAM_LLC, 22nm_SOTMRAM
cloud TPU: 3nm_SRAM, 3D_eDRAM_IWO_fast, 3D_FeFET_IWO, 3nm_STTMRAM_LLC, 3nm_SOTMRAM
memory options can be added or modified in ./NeuroSIM/main.cpp setting function

The integrated simulator will output system performance and store them in "TPU_benchmark.csv"