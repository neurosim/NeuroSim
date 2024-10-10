# NeuroSim

The NeuroSim framework was developed by [Prof. Shimeng Yu's group](https://shimeng.ece.gatech.edu/) (Georgia Institute of Technology). The model is made publicly available on a non-commercial basis. Copyright of the model is maintained by the developers, and the model is distributed under the terms of the [Creative Commons Attribution-NonCommercial 4.0 International Public License](http://creativecommons.org/licenses/by-nc/4.0/legalcode)

:star2: This repository includes all the versions of NeuroSim towards different nerual network applications. All versions are available in their own branch. Versions are divided into four categories, **2D Inference, 2D Training, 3D Inference, MLP**. Please read the following brief summary of each version and refer to the most suitable version. Detailed description can be found in the each branch's readme file and user manual.

## Versions and Key Features
**The latest version is modified by the lastest previous version.**
1. 2D Inference: DNN + NeuroSim evaluating DNN inference performance on 2D in-memory computing or near-memory computing architectures.
+ 2DInferenceV1.4: (1) Support 130nm~1nm technology node  (2) Add partial parallel mode (3) add XY bus as an alternative to H-tree interconnect
+ 2DInferenceV1.3: (1) Validate with real silicon data (2) Add synchronous and asynchronous mode (3) Update technology file for FinFET (4) Add level shifter for eNVM
+ 2DInferenceV1.2: (1) Calibrate temperature-related and layout features of FinFET technology (2) Add voltageSA-based multi-level SA and SAR ADC in addition to the original CurrentSA-based multi-level SA
+ 2DInferenceV1.1: (1) Multi-core architecture in buffer for high parallelism (2) Implement low-swing interconnect (3) Calibrate FinFET technology library (4) Add device non-ideal factors (5) Pipeline execution
+ 2DInferenceV1.0: (1) End-to-end benchmarking framework, (2) Support versatile memory technologies and various technology nodes (130nm~7nm), (3) Hierarchical simulation from chip level to array level

2. 2D Training: DNN + NeuroSim evaluating DNN inference performance and on-chip training performance on 2D in-memory computing or near-memory computing architectures.
+ 2DTrainingV1.2 (based on 2DInferenceV1.4): (1) Latest hardware evaluation (technology node, interconnect, synchronous and asynchronous computing, ...) ####TODO####
+ 2DTrainingV1.1 (based on 2DInferenceV1.2): (1) Include FeFET polarization during weight-update
+ 2DTrainingV1.0 (based on 2DInferenceV1.1): (1) Support on-chip training simulation and evaluation (2) Include more non-ideal properties of synaptic devices in programming compared to 2DInferenceV1.1

3. 3D Inference: DNN + NeuroSim evaluating DNN inference performance on heterogeneous 3D (H3D) and monolithic (M3D) in-memory computing or near-memory computing architectures.
+ 3DInferenceV1.1 (based on 2DInferenceV1.4): (1) Latest hardware evaluation (technology node, interconnect, synchronous and asynchronous computing, ...) ####TODO####
+ 3DInferenceV1.0 (based on 2DInferenceV1.3): (1) Support H3D and M3D evaluation

4. MLP Inference: MLP + NeuroSim evaluating DNN inference performance on 2D near-memory computing architectures.
+ MLPV3.0: (1) Add weight representation (-1,1) in addtion to the original (0,1) (2) Support more training optimization methods such as momentum method, Adagrad, RMSprop, Adam(3) Support parallel read-out for digital eNVMs (e.g. STT-MRAM) (4) Support hybrid precision synapses
+ MLPV2.0: (1) Support more device technologies (2) Optimize weight update scheme
+ MLPV1.0: (1) Support analog eNVM devices (RRAM or PCM)

Note: For the installation steps, file lists, release date, and additional details on the usage of this tool, please refer to the manual in the version branch.


## Acknowledgement
This research is supported by NSF CAREER award, NSF/SRC E2CDA program, the ASCENT center (SRC/DARPA JUMP 1.0) and the PRISM and CHIMES centers (SRC/DARPA JUMP 2.0).

## Developers
Developers: [Junmo Lee](mailto:junmolee@gatech.edu) :two_men_holding_hands: [James Read](mailto:jread6@gatech.edu) :couple: [Anni Lu](mailto:alu75@gatech.edu) :two_women_holding_hands: [Xiaochen Peng](mailto:xpeng76@gatech.edu) :two_women_holding_hands: [Shanshi Huang](mailto:shuang406@gatech.edu).

## Citations
If you use the tool or adapt the tool in your work or publication, you are required to cite the reference mentioned in the corresponding version branch.

## Contact
If you have logistic questions or comments on the model, please contact :man: [Prof. Shimeng Yu](mailto:shimeng.yu@ece.gatech.edu), and if you have technical questions or comments, please contact :man: [Junmo Lee](mailto:junmolee@gatech.edu) or :man: [James Read](mailto:jread6@gatech.edu) or :man: [Ming-Yen Lee](mlee838@gatech.edu)

:star2: Update
[2024.10.10] M3D version with the lastest hardware evaluation is avaiable. Check branch 3DInferenceV1.1/M3D for more details.

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