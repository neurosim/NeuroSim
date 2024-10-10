# 3D+NeuroSim V1.0

The DNN+NeuroSim framework was developed by [Prof. Shimeng Yu's group](https://shimeng.ece.gatech.edu/) (Georgia Institute of Technology). The model is made publicly available on a non-commercial basis. Copyright of the model is maintained by the developers, and the model is distributed under the terms of the [Creative Commons Attribution-NonCommercial 4.0 International Public License](http://creativecommons.org/licenses/by-nc/4.0/legalcode)

:star2: This is the released 3D+NeuroSim V1.0 (June 1, 2021) for the tool, and this version has **_improved following inference engine estimation_**:
```
1. Enabled electrical-thermal co-simulation of 3D integrated (monolithic and heterogeneous) CIM accelerators
2. Validate with real silicon data.
3. Add synchronous and asynchronous mode.
4. Update technology file for FinFET.
5. Add level shifter for eNVM.
```
:point_right: :point_right: :point_right: **For Monolithic-3D, in "Param.cpp", to switch mode:**
```
M3D = true;           // false: conventional 2D     // true: enable simulation for monolithic 3D integration
```
:point_right: :point_right: :point_right: **For Heterogeneous-3D, in "Param.cpp", to switch mode:**
```
H3D = true;           // false: conventional 2D     // true: enable simulation for hegerogeneous 3D integration

numMemTier = xxx;                 // user-defined number of memory tiers (on the top of logic tier)

deviceroadmapTop = xxx;           // device design options for top tiers (multi-tier memory arrays)
technodeTop = xxx;
featuresizeTop = xxx;

deviceroadmapBottom = xxx;        // device design options for bottom tier (other logic circuits)
technodeBottom = xxx;            
featuresizeBottom = xxx;

tsvPitch = xxx;                   // TSV pitch size
tsvRes = xxx;                     // TSV unit resistance
tsvCap = xxx;                     // TSV unit capacitance
```
:star2: This version has also added **_three default examples for quick start_**:
```
1. VGG8 on cifar10 
   8-bit "WAGE" mode pretrained model is uploaded to './log/VGG8.pth'
3. DenseNet40 on cifar10 
   8-bit "WAGE" mode pretrained model is uploaded to './log/DenseNet40.pth'
5. ResNet18 on imagenet 
   "FP" mode pretrained model is loaded from 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
```
:point_right: :point_right: :point_right: **To quickly start inference estimation of default models (skip training)**
```
python inference.py --dataset cifar10 --model VGG8 --mode WAGE
python inference.py --dataset cifar10 --model DenseNet40 --mode WAGE
python inference.py --dataset imagenet --model ResNet18 --mode FP
```

<br/>

**_For estimation of on-chip training accelerators, please visit released V2.1 [DNN+NeuroSim V2.1](https://github.com/neurosim/DNN_NeuroSim_V2.1)_**

In Pytorch/Tensorflow wrapper, users are able to define **_network structures, precision of synaptic weight and neural activation_**. With the integrated NeuroSim which takes real traces from wrapper, the framework can support hierarchical organization from device level to circuit level, to chip level and to algorithm level, enabling **_instruction-accurate evaluation on both accuracy and hardware performance of inference_**.

Developers: [Xiaochen Peng](mailto:xpeng76@gatech.edu) :two_women_holding_hands: [Shanshi Huang](mailto:shuang406@gatech.edu) :two_women_holding_hands: [Anni Lu](mailto:alu75@gatech.edu).

This research is supported by NSF CAREER award, NSF/SRC E2CDA program, and ASCENT, one of the SRC/DARPA JUMP centers.

If you use the tool or adapt the tool in your work or publication, you are required to cite the following reference:

**_X. Peng, W. Chakraborty, A. Kaul, W. Shim, M. S Bakir, S. Datta and S. Yu, ※[Benchmarking Monolithic 3D Integration for Compute-in-Memory Accelerators: Overcoming ADC Bottlenecks and Maintaining Scalability to 7nm or Beyond](https://ieeexplore.ieee.org/abstract/document/9372091), *§ IEEE International Electron Devices Meeting (IEDM)*, 2020._**

If you have logistic questions or comments on the model, please contact :man: [Prof. Shimeng Yu](mailto:shimeng.yu@ece.gatech.edu), and if you have technical questions or comments, please contact :woman: [Xiaochen Peng](mailto:xpeng76@gatech.edu) or :woman: [Shanshi Huang](mailto:shuang406@gatech.edu) or :woman: [Anni Lu](mailto:alu75@gatech.edu).


## File lists
1. Manual: `Documents/User Manual of 3D_NeuroSim_V1.0.pdf`
2. Framework for monolithic 3D integration: `Monolithic3D/inference.py` (to run Pytorch wrapper); `Monolithic3D/NeuroSim` (integrated NeuroSim core)
3. Framework for heterogeneous 3D integration: `Heterogeneous3D/inference.py` (to run Pytorch wrapper); `Heterogeneous3D/NeuroSim` (integrated NeuroSim core)


## Installation steps (Linux)
1. Get the tool from GitHub
```
git clone https://github.com/neurosim/3D_NeuroSim_V1.0.git
```

2. Go to the folder for either monolithic or heterogeneous 3D integration
```
cd Monolithic3D/
cd Heterogeneous3D/
```

3. Train the network to get the model for inference (can be skipped by using pretrained default models)

4. Compile the NeuroSim codes
```
make
```

5. Run Pytorch wrapper (integrated with NeuroSim)


For the usage of this tool, please refer to the manual.


## References related to this tool 
1. X. Peng, W. Chakraborty, A. Kaul, W. Shim, M. S Bakir, S. Datta and S. Yu, ※Benchmarking Monolithic 3D Integration for Compute-in-Memory Accelerators: Overcoming ADC Bottlenecks and Maintaining Scalability to 7nm or Beyond, *§ IEEE International Electron Devices Meeting (IEDM)*, 2020.
2. X. Peng, S. Huang, Y. Luo, X. Sun and S. Yu, ※DNN+NeuroSim: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators with Versatile Device Technologies, *§ IEEE International Electron Devices Meeting (IEDM)*, 2019.
3. X. Peng, R. Liu, S. Yu, ※Optimizing weight mapping and data flow for convolutional neural networks on RRAM based processing-in-memory architecture, *§ IEEE International Symposium on Circuits and Systems (ISCAS)*, 2019.
4. P.-Y. Chen, S. Yu, ※Technological benchmark of analog synaptic devices for neuro-inspired architectures, *§ IEEE Design & Test*, 2019.
5. P.-Y. Chen, X. Peng, S. Yu, ※NeuroSim: A circuit-level macro model for benchmarking neuro-inspired architectures in online learning, *§ IEEE Trans. CAD*, 2018.
6. X. Sun, S. Yin, X. Peng, R. Liu, J.-S. Seo, S. Yu, ※XNOR-RRAM: A scalable and parallel resistive synaptic architecture for binary neural networks,*§ ACM/IEEE Design, Automation & Test in Europe Conference (DATE)*, 2018.
7. P.-Y. Chen, X. Peng, S. Yu, ※NeuroSim+: An integrated device-to-algorithm framework for benchmarking synaptic devices and array architectures, *§ IEEE International Electron Devices Meeting (IEDM)*, 2017.
8. P.-Y. Chen, S. Yu, ※Partition SRAM and RRAM based synaptic arrays for neuro-inspired computing,*§ IEEE International Symposium on Circuits and Systems (ISCAS)*, 2016.
9. P.-Y. Chen, D. Kadetotad, Z. Xu, A. Mohanty, B. Lin, J. Ye, S. Vrudhula, J.-S. Seo, Y. Cao, S. Yu, ※Technology-design co-optimization of resistive cross-point array for accelerating learning algorithms on chip,*§ IEEE Design, Automation & Test in Europe (DATE)*, 2015.
10. S. Wu, et al., ※Training and inference with integers in deep neural networks,*§ arXiv: 1802.04680*, 2018.
11. github.com/boluoweifenda/WAGE
12. github.com/stevenygd/WAGE.pytorch
13. github.com/aaron-xichen/pytorch-playground
