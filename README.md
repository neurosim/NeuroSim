# DNN+NeuroSim V1.5

The DNN+NeuroSim framework was developed by [Prof. Shimeng Yu's group](https://shimeng.ece.gatech.edu/) (Georgia Institute of Technology). The model is made publicly available on a non-commercial basis. Copyright of the model is maintained by the developers, and the model is distributed under the terms of the [Creative Commons Attribution-NonCommercial 4.0 International Public License](http://creativecommons.org/licenses/by-nc/4.0/legalcode)

## What's New in Version 1.5 (May 1, 2025)

This version introduces significant improvements to the inference engine for compute-in-memory accelerators:

### 1. Enhanced Neural Network Support
- **Custom Model Integration**: Design and use your own neural network architectures from PyTorch, PyTorch Hub, or HuggingFace
- **Model Quantization**: Seamless integration with NVIDIA TensorRT for post-training quantization
- **Pre-trained Models**: Includes pre-trained models (VGG8 for CIFAR-10, ResNet18 for CIFAR-100). Models trained on ImageNet dataset (ResNet-50 and Swin-T) are downloaded from PyTorch hub. Currently supported models are ResNet18 to ResNet151 and Swin-T. Users can easily add support for more models by following the code in quantize.py
- **Transformer Support**: Added capability to simulate transformer architectures like Swin-T

### 2. Flexible Noise Modeling
- **Comprehensive Device Non-idealities**: Model device variation, stuck-at-faults, retention, and more
- **Statistical Noise Models**: Import noise profiles from SPICE simulations or real silicon measurements
- **Custom Memory States**: Define detailed memory cell characteristics through CSV configuration

### 3. New Memory Technology Support
- **Non-volatile Capacitive Memory (nvCap-CIM)**: Full support for charge-based computation
- Configure nvCap parameters in `Param.cpp`:
  ```
  memcelltype = 4;
  accesstype = 4;
  chargeDelay = 5e-9;
  // Other parameters (lines 374 - 382)
  ```

### 4. Performance Improvements
- Up to 6.5x faster runtime compared to V1.5
- Optimized PyTorch integration for accelerated computation

## Installation Instructions
1. Download the tool from Github
```bash
git clone https://github.com/neurosim/NeuroSim
cd NeuroSim
git checkout 2DInferenceV1.5
```
2. Set the environment name (first line) and install folder prefix (last line) in environment.yml

3. Create conda environment and activate the environment 
```bash
conda env create -f environment.yml
conda activate neurosim
```

4. Install TensorRT
```bash
cd pytorch-quantization
pip install -e .
```

5. Compile hardware evaluation code
```bash
cd ../NeuroSIM
make
```

## Dataset Preparation
For ImageNet dataset, use the provided script to prepare the data:
```bash
mkdir -p datasets/imagenet
cd datasets/imagenet
# Copy the get_imagenet.sh script to this directory
bash get_imagenet.sh
```
This script will download and organize the ImageNet dataset in the correct format. When running inference, provide this path with the `--data_path` parameter.

## Key Files and Components

### Core Components
- **inference.py**: Main entry point that runs the inference simulation with hardware modeling
  - Controls simulation parameters (hardware configuration, noise models, quantization settings)
  - Provides command-line interface for all simulation options
  - Manages evaluation and hardware performance analysis workflow

- **quantize.py**: Handles neural network model loading and quantization
  - Loads models from local files, PyTorch Hub, or HuggingFace
  - Integrates with TensorRT for post-training quantization
  - Maps neural network operations to compute-in-memory equivalents
  - Example framework for supporting custom model architectures

- **NeuroSIM/**: C++ hardware simulation backend
  - Detailed circuit-level modeling for power, performance, area estimation
  - Component library for various memory technologies and peripheral circuits
  - Technology scaling parameters for different nodes

- **pytorch-quantization/pytorch_quantization/cim**: NeuroSim integration with TensorRT 
  - Custom CIM (Compute-In-Memory) module implementation
  - Hardware-aware quantization for memory array operation

## Running Examples

```bash
# Example 1: VGG8 on CIFAR-10 with resistive memory
python inference.py --dataset cifar10 --model vgg8 --hardware 1 --bitcell 1 \
    --sub_array "[128,128]" --mem_type "resistive" --mem_states_file "mem_states.csv" \
    --adc_precision 7

# Example 2: ResNet18 on CIFAR-100 with capacitive memory
python inference.py --dataset cifar100 --model resnet18 --hardware 1 --bitcell 1 \
    --sub_array "[128,128]" --mem_type "capacitive" --mem_states_file "" \
    --off_state 1e-15 --on_state 1e-14

# Example 3: ResNet50 on ImageNet with capacitive memory and read noise
python inference.py --dataset imagenet --model resnet50 --hardware 1 --bitcell 1 \
    --sub_array "[128,128]" --mem_type "capacitive" --mem_states_file "" \
    --read_noise 0.1
```

## Parameter Reference

### Memory Configuration Parameters
| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `--mem_type` | Memory cell type (resistive/capacitive) | "resistive", "capacitive" |
| `--bitcell` | Cell precision (bits per cell) | 1, 2, 4 |
| `--sub_array` | Size of subArray (e.g. [128, 128]) | "[128, 128]", "[64, 64]" |
| `--parallel_read` | Number of rows read in parallel | 128, 64, 32 |

### Memory State Parameters
| Parameter | Description | Usage |
|-----------|-------------|-------|
| `--mem_states_file` | Path to CSV with memory state characteristics | "mem_states.csv" |
| `--off_state` | Device off state (conductance (S) or capacitance (F)) | 1e-15 |
| `--on_state` | Device on conductance/capacitance | 25e-15 |

**Note**: For `--mem_states_file`, you can:
- Provide a path to a CSV file with detailed memory state definitions
- Set to empty string (`""`) to use the `off_state` and `on_state` parameters instead

### Noise Modeling Parameters
| Parameter | Description | Usage |
|-----------|-------------|-------|
| `--read_noise` | Standard deviation of conductance read noise | 0.0, 0.1 |
| `--output_noise` | Standard deviation of output voltage noise | 0.0, 0.1 |
| `--output_noise_file` | Path to CSV with output noise statistics | "output_noise.csv" |

**Note**: For `--output_noise`:  
- Set to a positive value for uniform noise distribution
- Set to `-1` to load the output statistics from the file specified by `--output_noise_file`

### Reliability Parameters
| Parameter | Description | Usage |
|-----------|-------------|-------|
| `--t` | Retention time | 1, 1000, 1e6 |
| `--v` | Drift coefficient | 0.00, 0.01 |
| `--detect` | Drift direction (0: random, 1: fixed) | 0, 1 |
| `--target` | Drift target for fixed direction | 0.0, 0.5, 1.0 |
| `--rate_stuck_0` | Rate of cells stuck at 0 | 0.00, 0.01 |
| `--rate_stuck_1` | Rate of cells stuck at 1 | 0.00, 0.01 |

### Quantization Parameters
| Parameter | Description | Usage |
|-----------|-------------|-------|
| `--weight_precision` | Bits for weight quantization | 8, 6, 4 |
| `--input_precision` | Bits for input quantization | 8, 6, 4 |
| `--dac_precision` | DAC precision | 1, 4, 8 |
| `--adc_precision` | ADC precision | 7, 5, 3 |
| `--input_calib_method` | Input calibration method | "max", "histogram" |
| `--weight_calib_method` | Weight calibration method | "max", "histogram" |

**Note: dac_precision is only supported in the software backbone (inference accuracy simulations). All PPA simulations will use bit-serial mode regardless of what dac_precision is set to.**

## Adding Custom Models

To add your own neural network model:

1. Create a model definition file (follow examples in `models/vgg.py` or `models/resnet.py`)
2. Train your model using PyTorch and save the weights
3. Add your model to `quantize.py` by following the templates for existing models
4. For models from PyTorch Hub or HuggingFace, use the loading examples in `quantize.py`
5. Specify `model` parameter when running `inference.py`

## Citation

If you use the tool or adapt the tool in your work or publication, you are required to cite the following reference:

**_J. Read, M-Y. Lee, W-H. Huang, Y-C. Luo, A. Lu, S. Yu, ※[NeuroSim V1.5: Improved Software Backbone for Benchmarking Compute-in-Memory Accelerators with Device and Circuit-level Non-idealities](https://arxiv.org/abs/2505.02314), *§ arXiv, 2025._**

**_J. Lee, A. Lu, W. Li, S. Yu, ※[NeuroSim V1.4: Extending Technology Support for Digital Compute-in-Memory Toward 1nm Node](https://ieeexplore.ieee.org/document/10443264), *§ IEEE Transactions on Circuits and Systems I: Regular Papers*, 2024._**

**_X. Peng, S. Huang, Y. Luo, X. Sun and S. Yu, ※[DNN+NeuroSim: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators with Versatile Device Technologies](https://ieeexplore-ieee-org.prx.library.gatech.edu/document/8993491), *§ IEEE International Electron Devices Meeting (IEDM)*, 2019._**

## Update
:star2: [2025.06.27] Image normalization of ImageNet dataset is updated for Swin Transformer following PyTorch Libraries. Software top-1 accuracy: 81.7% (previous: 80.1%)

## Contributors and Support

Developers: [James Read](mailto:jread6@gatech.edu) :two_men_holding_hands: [Ming-Yen Lee](mailto:mlee838@gatech.edu) :two_men_holding_hands: [Junmo Lee](mailto:junmolee@gatech.edu) :couple: [Anni Lu](mailto:alu75@gatech.edu) :two_women_holding_hands: [Xiaochen Peng](mailto:xpeng76@gatech.edu) :two_women_holding_hands: [Shanshi Huang](mailto:shuang406@gatech.edu).

This research is supported by NSF CAREER award, NSF/SRC E2CDA program, the ASCENT center (SRC/DARPA JUMP 1.0) and the PRISM and CHIMES centers (SRC/DARPA JUMP 2.0).

If you have logistic questions or comments on the model, please contact :man: [Prof. Shimeng Yu](mailto:shimeng.yu@ece.gatech.edu), and if you have technical questions or comments, please contact :man: [James Read](mailto:jread6@gatech.edu) or :man: [Ming-Yen Lee](mailto:mlee838@gatech.edu) :man: [Junmo Lee](mailto:junmolee@gatech.edu).

## References related to this tool
1. J. Read, M-Y. Lee, W-H. Huang, Y-C. Luo, A. Lu, S. Yu, "NeuroSim V1.5: Improved Software Backbone for Benchmarking Compute-in-Memory Accelerators with Device and Circuit-level Non-idealities," arXiv, 2025.
2. Y. -C. Luo, J. Read, A. Lu and S. Yu, "A Cross-layer Framework for Design Space and Variation Analysis of Non-Volatile Ferroelectric Capacitor-Based Compute-in-Memory Accelerators," 2024 29th Asia and South Pacific Design Automation Conference (ASP-DAC), 2024.
3. J. Lee, A. Lu, W. Li, S. Yu, ※NeuroSim V1. 4: Extending Technology Support for Digital Compute-in-Memory Toward 1nm Node, *§ IEEE Transactions on Circuits and Systems I: Regular Papers, 2024.
4. A. Lu, X. Peng, W. Li, H. Jiang, S. Yu, ※NeuroSim simulator for compute-in-memory hardware accelerator: validation and benchmark, *§ Frontiers in Artificial Intelligence, vol. 4, 659060, 2021.
5. X. Peng, S. Huang, Y. Luo, X. Sun and S. Yu, ※DNN+NeuroSim: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators with Versatile Device Technologies, *§ IEEE International Electron Devices Meeting (IEDM)*, 2019.
6. X. Peng, R. Liu, S. Yu, ※Optimizing weight mapping and data flow for convolutional neural networks on RRAM based processing-in-memory architecture, *§ IEEE International Symposium on Circuits and Systems (ISCAS)*, 2019.
7. P.-Y. Chen, S. Yu, ※Technological benchmark of analog synaptic devices for neuro-inspired architectures, *§ IEEE Design & Test*, 2019.
8. P.-Y. Chen, X. Peng, S. Yu, ※NeuroSim: A circuit-level macro model for benchmarking neuro-inspired architectures in online learning, *§ IEEE Trans. CAD*, 2018.
9. X. Sun, S. Yin, X. Peng, R. Liu, J.-S. Seo, S. Yu, ※XNOR-RRAM: A scalable and parallel resistive synaptic architecture for binary neural networks,*§ ACM/IEEE Design, Automation & Test in Europe Conference (DATE)*, 2018.
10. P.-Y. Chen, X. Peng, S. Yu, ※NeuroSim+: An integrated device-to-algorithm framework for benchmarking synaptic devices and array architectures, *§ IEEE International Electron Devices Meeting (IEDM)*, 2017.
11. P.-Y. Chen, S. Yu, ※Partition SRAM and RRAM based synaptic arrays for neuro-inspired computing,*§ IEEE International Symposium on Circuits and Systems (ISCAS)*, 2016.
12. P.-Y. Chen, D. Kadetotad, Z. Xu, A. Mohanty, B. Lin, J. Ye, S. Vrudhula, J.-S. Seo, Y. Cao, S. Yu, ※Technology-design co-optimization of resistive cross-point array for accelerating learning algorithms on chip,*§ IEEE Design, Automation & Test in Europe (DATE)*, 2015.
13. S. Wu, et al., ※Training and inference with integers in deep neural networks,*§ arXiv: 1802.04680*, 2018.
14. github.com/boluoweifenda/WAGE
15. github.com/stevenygd/WAGE.pytorch
16. github.com/aaron-xichen/pytorch-playground
