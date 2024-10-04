# DNN+NeuroSim V1.4

The DNN+NeuroSim framework was developed by [Prof. Shimeng Yu's group](https://shimeng.ece.gatech.edu/) (Georgia Institute of Technology). The model is made publicly available on a non-commercial basis. Copyright of the model is maintained by the developers, and the model is distributed under the terms of the [Creative Commons Attribution-NonCommercial 4.0 International Public License](http://creativecommons.org/licenses/by-nc/4.0/legalcode)

:star2: This is the released version 1.4 (August 1, 2023) for the tool, and this version has **_improved following inference engine estimation_**:

## 1. Support for technology scaling down to 1nm node in the hardware estimation framework (C++ code).
```
The following is a list of the supported nodes with key features:

130nm Same as V1.3 (PTM Model)
90nm Same as V1.3 (PTM Model)
65nm Same as V1.3 (PTM Model)
45nm Same as V1.3 (PTM Model)
32nm Same as V1.3 (PTM Model)
22nm Same as V1.3 (PTM Model)
14nm FinFET, Fin number=4 (per each NMOS/PMOS)
10nm FinFET, Fin number=3 (per each NMOS/PMOS)
7nm FinFET, Fin number=2 (per each NMOS/PMOS)
5nm FinFET, Fin number=2 (per each NMOS/PMOS)
3nm FinFET, Fin number=2 (per each NMOS/PMOS)
2nm Nanosheet (GAA), Nanosheet number=3 (per each NMOS/PMOS), Backside power rail
1nm Nanosheet (GAA), Nanosheet number=4 (per each NMOS/PMOS), Backside power rail, Dielectric wall separation, CFET design for SRAM
```

### To select a technology node:
```
modify the "technnode" parameter (line 127 in Param.cpp) to match the desired case
For example technnode = 22 corresponds to a technode of 22nm and technnode = 14 corresponds to 14nm.

For additional details about the device parameters used in NeuroSim, refer to section 7 of the V1.4 manual.
```

## 2. Add partial parallel mode in python wrapper (for single-level cells only) and C++ code for hardware estimation.

### To enable partial parallel mode
```
Specify the parameter "parallelRead" when running the python wrapper.
Partial parallel mode will be enabled in both the python wrapper and C++ code.

--parallelRead N

Where N is the desired number of rows activated in parallel and N <= sub-array size.
```

## 3. XY Bus as an alternative to H-tree interconnect.

### In "Param.cpp", to switch interconnect mode:
```
globalBusType = false;		// false: X-Y Bus      // true: H-Tree
```

## Installation steps (Linux + Anaconda/Miniconda)
We recommend using anaconda package manager to install PyTorch.

This version supports the recently released PyTorch 2.0

We have currently tested the following operating systems and drivers:

(1) 
Red Hat 8.8 (Ootpa)
gcc: v8.5.0
glibc: v2.28
NVIDIA Driver Version: 535.54.03
CUDA Version: 12.2

(2)
Ubuntu 20.04
gcc: v9.4.0
glibc: v2.31
NVIDIA Driver Version: 525.60.13
CUDA Version: 12.0

### 1. Download Anaconda/Miniconda: https://docs.conda.io/en/latest/miniconda.html
### 2. Follow install instructions: https://docs.conda.io/en/latest/miniconda.html#installing

### 3. Get the tool from GitHub
```
git clone https://github.com/neurosim/DNN_NeuroSim_V1.4.git
cd DNN_NeuroSim_V1.4
```

### 4. Create a conda environment

```
conda create --name neurosim
```

### 5. Activate neurosim environment

```
conda activate neurosim
```

### 6. Download and install PyTorch packages

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

### 7. Pick a network architecture. The following have been pre-trained and provided with NeuroSim.
```
1. VGG8 on cifar10 
   8-bit "WAGE" mode pretrained model is uploaded to './log/VGG8.pth'
2. DenseNet40 on cifar10 
   8-bit "WAGE" mode pretrained model is uploaded to './log/DenseNet40.pth'
3. ResNet18 on imagenet 
   "FP" mode pretrained model is loaded from 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
```

### 8. (Optional) Train the network to get the model for inference

### 9. Compile the NeuroSim C++ code
```
cd Inference_pytorch/NeuroSIM
make
```

### 10. Run Pytorch/Tensorflow wrapper (integrated with NeuroSim). The following are some examples with arguments.

```
cd ..

python inference.py --dataset cifar10 --model VGG8 --mode WAGE --inference 1 --cellBit 1 --subArray 128 --parallelRead 64
python inference.py --dataset cifar10 --model DenseNet40 --mode WAGE --inference 1 --cellBit 2 --ADCprecision 6
python inference.py --dataset imagenet --model ResNet18 --mode FP --inference 1 --onoffratio 100
```

<br/>

**_For estimation of on-chip training accelerators, please visit released V2.1 [DNN+NeuroSim V2.1](https://github.com/neurosim/DNN_NeuroSim_V2.1)_**
```
NOTE: the on-chip training framework has not yet been updated to support the features released in this version (DNN+NeuroSim V1.4).

We plan to support the technology scaling to 1nm, partial parallel mode and the XY bus in a future update.  
```
In Pytorch/Tensorflow wrapper, users are able to define **_network structures, precision of synaptic weight and neural activation_**. With the integrated NeuroSim which takes real traces from wrapper, the framework can support hierarchical organization from device level to circuit level, to chip level and to algorithm level, enabling **_instruction-accurate evaluation on both accuracy and hardware performance of inference_**.

Developers: [Junmo Lee](mailto:junmolee@gatech.edu) :two_men_holding_hands: [James Read](mailto:jread6@gatech.edu) :couple: [Anni Lu](mailto:alu75@gatech.edu) :two_women_holding_hands: [Xiaochen Peng](mailto:xpeng76@gatech.edu) :two_women_holding_hands: [Shanshi Huang](mailto:shuang406@gatech.edu).

This research is supported by NSF CAREER award, NSF/SRC E2CDA program, the ASCENT center (SRC/DARPA JUMP 1.0) and the PRISM and CHIMES centers (SRC/DARPA JUMP 2.0).

If you use the tool or adapt the tool in your work or publication, you are required to cite the following reference:

**_X. Peng, S. Huang, Y. Luo, X. Sun and S. Yu, ※[DNN+NeuroSim: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators with Versatile Device Technologies](https://ieeexplore-ieee-org.prx.library.gatech.edu/document/8993491), *§ IEEE International Electron Devices Meeting (IEDM)*, 2019._**

If you have logistic questions or comments on the model, please contact :man: [Prof. Shimeng Yu](mailto:shimeng.yu@ece.gatech.edu), and if you have technical questions or comments, please contact :man: [Junmo Lee](mailto:junmolee@gatech.edu) or :man: [James Read](mailto:jread6@gatech.edu) or :woman: [Anni Lu](mailto:alu75@gatech.edu).


## File lists
1. Manual: `Documents/DNN NeuroSim V1.4 Manual.pdf`
2. DNN_NeuroSim wrapped by Pytorch: 'Inference_pytorch'
3. NeuroSim under Pytorch Inference: 'Inference_pytorch/NeuroSIM'



For additional details on the usage of this tool, please refer to the manual.


## References related to this tool 
1. J. Lee, A. Lu, W. Li, S. Yu, ※NeuroSim V1. 4: Extending Technology Support for Digital Compute-in-Memory Toward 1nm Node, *§ IEEE Transactions on Circuits and Systems I: Regular Papers, 2024.
2. A. Lu, X. Peng, W. Li, H. Jiang, S. Yu, ※NeuroSim simulator for compute-in-memory hardware accelerator: validation and benchmark, *§ Frontiers in Artificial Intelligence, vol. 4, 659060, 2021.
3. X. Peng, S. Huang, Y. Luo, X. Sun and S. Yu, ※DNN+NeuroSim: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators with Versatile Device Technologies, *§ IEEE International Electron Devices Meeting (IEDM)*, 2019.
4. X. Peng, R. Liu, S. Yu, ※Optimizing weight mapping and data flow for convolutional neural networks on RRAM based processing-in-memory architecture, *§ IEEE International Symposium on Circuits and Systems (ISCAS)*, 2019.
5. P.-Y. Chen, S. Yu, ※Technological benchmark of analog synaptic devices for neuro-inspired architectures, *§ IEEE Design & Test*, 2019.
6. P.-Y. Chen, X. Peng, S. Yu, ※NeuroSim: A circuit-level macro model for benchmarking neuro-inspired architectures in online learning, *§ IEEE Trans. CAD*, 2018.
7. X. Sun, S. Yin, X. Peng, R. Liu, J.-S. Seo, S. Yu, ※XNOR-RRAM: A scalable and parallel resistive synaptic architecture for binary neural networks,*§ ACM/IEEE Design, Automation & Test in Europe Conference (DATE)*, 2018.
8. P.-Y. Chen, X. Peng, S. Yu, ※NeuroSim+: An integrated device-to-algorithm framework for benchmarking synaptic devices and array architectures, *§ IEEE International Electron Devices Meeting (IEDM)*, 2017.
9. P.-Y. Chen, S. Yu, ※Partition SRAM and RRAM based synaptic arrays for neuro-inspired computing,*§ IEEE International Symposium on Circuits and Systems (ISCAS)*, 2016.
10. P.-Y. Chen, D. Kadetotad, Z. Xu, A. Mohanty, B. Lin, J. Ye, S. Vrudhula, J.-S. Seo, Y. Cao, S. Yu, ※Technology-design co-optimization of resistive cross-point array for accelerating learning algorithms on chip,*§ IEEE Design, Automation & Test in Europe (DATE)*, 2015.
11. S. Wu, et al., ※Training and inference with integers in deep neural networks,*§ arXiv: 1802.04680*, 2018.
12. github.com/boluoweifenda/WAGE
13. github.com/stevenygd/WAGE.pytorch
14. github.com/aaron-xichen/pytorch-playground
