# DNN+NeuroSim Digital Systolic Array V1.0

The DNN+Digital Systolic Array+NeuroSim framework was developed by [Prof. Shimeng Yu's group](https://shimeng.ece.gatech.edu/) (Georgia Institute of Technology). The model is made publicly available on a non-commercial basis. Copyright of the model is maintained by the developers, and the model is distributed under the terms of the [Creative Commons Attribution-NonCommercial 4.0 International Public License](http://creativecommons.org/licenses/by-nc/4.0/legalcode)

:star2: This is the DNN+Digital Systolic Array+NeuroSim V1.0 (Nov, 2024) for the tool, where VGG8 and ResNet18 model are supported for DNN. Memory access times are generated by [Timeloop](https://timeloop.csail.mit.edu/) (from Massachusetts Institute of Technology), and NeuroSim wraps the DNN algorithms and hardware to generate the hardware PPA report. This version provides a ready-made testbench with memory access times for VGG8 and ResNet18, which can be found in the /traces folder. To use a custom testbench, please refer to the data format of the files in /traces folder and adapt it to incorporate your data. In addition, this version mimics Google's TPU V1.0 with systolic arrays and on-chip buffers, which we describe as a TPU-like digital systolic array. This version has **updated the hardware evaluation to the latest version (V1.4)**. (For the features of the latest hardware evaluation, please check the branches of DNN_NeuroSim_V1.4 for more details.)

## Tutorials
Please check the user manual `documents/User Manual of NeuroSim Digital Systolic Array Simulator.pdf` to run the evaluation.

## Installation steps (Linux + Anaconda/Miniconda)

We have currently tested the following operating systems and drivers:

(1) 
Red Hat 8.8 (Ootpa)
gcc: v8.5.0
glibc: v2.28
NVIDIA Driver Version: 535.54.03
CUDA Version: 12.2

### 1. Download Anaconda/Miniconda: https://docs.conda.io/en/latest/miniconda.html
### 2. Follow install instructions: https://docs.conda.io/en/latest/miniconda.html#installing

### 3. Get the tool from GitHub
```
git clone -b 2DInferenceDigitalSystolicArrayV1.0 --single-branch https://github.com/neurosim/NeuroSim.git
cd NeuroSim
```
### 4. Create a conda environment

```
conda create --name neurosim
```

### 5. Activate neurosim environment

```
conda activate neurosim
```

### 6. Compile the NeuroSim C++ code
```
cd Inference_pytorch/NeuroSIM
make clean
make
```

### 7. Run the wrapper (integrated with NeuroSim). The following are some examples with arguments.

```
cd ..

python main.py --tputype edge --model VGG8 
python main.py --tputype edge --model ResNet18 
python main.py --tputype cloud --model VGG8 

```

<br/>

**_For estimation of on-chip training accelerators, please visit released V2.1 [DNN+NeuroSim V2.1](https://github.com/neurosim/DNN_NeuroSim_V2.1)_**


Developers: Anni Lu :couple: Yandong Luo :two_men_holding_hands: [Junmo Lee](mailto:junmolee@gatech.edu) :two_women_holding_hands: Xiaochen Peng :two_women_holding_hands: Shanshi Huang.

This research is supported by NSF CAREER award, NSF/SRC E2CDA program, the ASCENT center (SRC/DARPA JUMP 1.0) and the PRISM and CHIMES centers (SRC/DARPA JUMP 2.0).

If you use the tool or adapt the tool in your work or publication, you are required to cite the following reference:

**_A. Lu, J. Lee, T. H. Kim, M. Karim, R. S. Park, H. Simka, S. Yu, ※[High-speed emerging memories for AI hardware accelerators](https://www.nature.com/articles/s44287-023-00002-9). *§Nature Reviews Electrical Engineering*, 1(1), 24-34, 2024._**

If you have logistic questions or comments on the model, please contact :man: [Prof. Shimeng Yu](mailto:shimeng.yu@ece.gatech.edu), and if you have technical questions or comments, please contact :man: [Junmo Lee](mailto:junmolee@gatech.edu) or :man: [James Read](mailto:jread6@gatech.edu) or :man: [Ming-Yen Lee](mailto:mlee838@gatech.edu).


## File lists
1. Manual: `documents/User Manual of NeuroSim Digital Systolic Array Simulator.pdf`, `documents/readme.txt`
2. DNN with digital systolic arrays wrapped by Pytorch: `main.py`, `input.py`, `evaluate.py`, `hardware.py`, `parameters.py`, `utils.py`
3. NeuroSim under Pytorch Inference: '/NeuroSIM'
4. Memory and buffer access count tables (from Timeloop): '/traces'

For additional details on the usage of this tool, please refer to the manual.

## References related to this tool 
1. A. Lu, J. Lee, T. H. Kim, M. Karim, R. S. Park, H. Simka, S. Yu, ※High-speed emerging memories for AI hardware accelerators. *§Nature Reviews Electrical Engineering*, 1(1), 24-34, 2024.
1. J. Lee, A. Lu, W. Li, S. Yu, ※NeuroSim V1.4: Extending Technology Support for Digital Compute-in-Memory Toward 1nm Node, *§ IEEE Transactions on Circuits and Systems I: Regular Papers*, 2024.
1. A. Parashar, P. Raina; Y. S. Shao; Y.-H. Chen; V. A. Ying; A. Mukkara, ※Timeloop: A systematic approach to dnn accelerator evaluation. *§2019 IEEE international symposium on performance analysis of systems and software (ISPASS)*. IEEE, 2019.
2. A. Lu, X. Peng, W. Li, H. Jiang, S. Yu, ※NeuroSim simulator for compute-in-memory hardware accelerator: validation and benchmark, *§ Frontiers in Artificial Intelligence, vol. 4, 659060*, 2021.
3. H. Ye, J. Gomez, W. Chakraborty, S. Spetalnick, S. Dutta, K. Ni, A. Raychowdhury, S. Datta, ※Double-gate W-doped amorphous indium oxide transistors for monolithic 3D capacitorless gain cell eDRAM. *§ IEEE International Electron Devices Meeting (IEDM)*, 2020.
4. X. Peng, S. Huang, Y. Luo, X. Sun and S. Yu, ※DNN+NeuroSim: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators with Versatile Device Technologies, *§ IEEE International Electron Devices Meeting (IEDM)*, 2019.
5. X. Peng, R. Liu, S. Yu, ※Optimizing weight mapping and data flow for convolutional neural networks on RRAM based processing-in-memory architecture, *§ IEEE International Symposium on Circuits and Systems (ISCAS)*, 2019.
6. P.-Y. Chen, S. Yu, ※Technological benchmark of analog synaptic devices for neuro-inspired architectures, *§ IEEE Design & Test*, 2019.
7. P.-Y. Chen, X. Peng, S. Yu, ※NeuroSim: A circuit-level macro model for benchmarking neuro-inspired architectures in online learning, *§ IEEE Trans. CAD*, 2018.
8. X. Sun, S. Yin, X. Peng, R. Liu, J.-S. Seo, S. Yu, ※XNOR-RRAM: A scalable and parallel resistive synaptic architecture for binary neural networks,*§ ACM/IEEE Design, Automation & Test in Europe Conference (DATE)*, 2018.
9. P.-Y. Chen, X. Peng, S. Yu, ※NeuroSim+: An integrated device-to-algorithm framework for benchmarking synaptic devices and array architectures, *§ IEEE International Electron Devices Meeting (IEDM)*, 2017.
10. P.-Y. Chen, S. Yu, ※Partition SRAM and RRAM based synaptic arrays for neuro-inspired computing,*§ IEEE International Symposium on Circuits and Systems (ISCAS)*, 2016.
11. P.-Y. Chen, D. Kadetotad, Z. Xu, A. Mohanty, B. Lin, J. Ye, S. Vrudhula, J.-S. Seo, Y. Cao, S. Yu, ※Technology-design co-optimization of resistive cross-point array for accelerating learning algorithms on chip,*§ IEEE Design, Automation & Test in Europe (DATE)*, 2015.
12. S. Wu, et al., ※Training and inference with integers in deep neural networks,*§ arXiv: 1802.04680*, 2018.
13. github.com/boluoweifenda/WAGE
14. github.com/stevenygd/WAGE.pytorch
15. github.com/aaron-xichen/pytorch-playground
