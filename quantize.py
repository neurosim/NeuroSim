import time
import torch
import torch.utils.data
from tqdm import tqdm
from torchvision import models
from models import vgg, resnet

import pytorch_quantization.cim.modules.macro as macro
import pytorch_quantization.nn as quant_nn
import pytorch_quantization.calib as calib
from pytorch_quantization.tensor_quant import QuantDescriptor
import pytorch_quantization.quant_modules as quant_modules
from pytorch_quantization import cim

from collections import namedtuple

# Definition of the named tuple that is used to store mapping of the quantized modules
_quant_entry = namedtuple('quant_entry', 'orig_mod mod_name replace_mod')

# Global member of the file that contains the mapping of quantized modules
cim_quant_map = [_quant_entry(torch.nn, "Conv2d", cim.CIMConv2d),
                 _quant_entry(torch.nn, "Linear", cim.CIMLinear),
                 _quant_entry(torch.nn, "MaxPool2d", cim.CIMMaxPool2d),
                 _quant_entry(torch.nn, "AvgPool2d", cim.CIMAvgPool2d),
                 _quant_entry(torch.nn, "AdaptiveAvgPool2d", cim.CIMAdaptiveAvgPool2d)]


# Global member of the file that contains the mapping of quantized modules
quant_map = [_quant_entry(torch.nn, "Conv2d", quant_nn.QuantConv2d),
             _quant_entry(torch.nn, "Linear", quant_nn.QuantLinear),
             _quant_entry(torch.nn, "MaxPool2d", quant_nn.QuantMaxPool2d),
             _quant_entry(torch.nn, "AvgPool2d", quant_nn.QuantAvgPool2d),
             _quant_entry(torch.nn, "AdaptiveAvgPool2d", cim.CIMAdaptiveAvgPool2d)]



def quantize_model(args, criterion, data_loader, data_loader_test):
    # Step 1: Initialize quantization modules and CIM arguments

    # Turn off hardware mode for quantization
    hardware = args.hardware
    args.hardware = False

    # TODO: Support more layer types
    quant_modules.initialize(float_module_list=['Conv2d', 'Linear', 
                                                'AvgPool2d', 'MaxPool2d', 
                                                'AdaptiveAvgPool2d'], custom_quant_modules=cim_quant_map)

    # Set axis to none for histogram mode
    if args.input_calib_method == 'histogram':
        args.input_axis = None
    if args.weight_calib_method == 'histogram':
        args.weight_axis = None
    if args.adc_calib_method == 'histogram':
        args.adc_axis = None

    # Turn on fake quant to calibrate with GPU
    args.fake_quant = True

    input_desc =     QuantDescriptor(calib_method=args.input_calib_method, num_bits=args.input_precision, 
                                     fake_quant=args.fake_quant, axis=args.input_axis, unsigned=False)
    
    weight_desc =    QuantDescriptor(calib_method=args.weight_calib_method, num_bits=args.weight_precision, 
                                     fake_quant=args.fake_quant, axis=args.weight_axis, unsigned=False)
    
    adc_quant_desc = QuantDescriptor(calib_method=args.adc_calib_method, num_bits=args.adc_precision, 
                                     fake_quant=args.fake_quant, axis=args.adc_axis, unsigned=True)

    cim.CIMLinear.set_default_quant_desc_input( input_desc)
    cim.CIMLinear.set_default_quant_desc_weight(weight_desc)
    cim.CIMLinear.set_default_quant_desc_adc(adc_quant_desc)
    cim.CIMLinear.set_default_cim_args(args)

    cim.CIMConv2d.set_default_quant_desc_input( input_desc)
    cim.CIMConv2d.set_default_quant_desc_weight(weight_desc)
    cim.CIMConv2d.set_default_quant_desc_adc(adc_quant_desc)
    cim.CIMConv2d.set_default_cim_args(args)

    cim.CIMMaxPool2d.set_default_quant_desc_input( input_desc)
    cim.CIMMaxPool2d.set_default_cim_args(args)

    cim.CIMAvgPool2d.set_default_cim_args(args)
    cim.CIMAvgPool2d.set_default_quant_desc_input( input_desc)

    cim.CIMAdaptiveAvgPool2d.set_default_cim_args(args)
    cim.CIMAdaptiveAvgPool2d.set_default_quant_desc_input( input_desc)

    # Step 2: Load pretrained model
    saved_model = args.model_path + args.model + '_' + args.dataset + '.pth'

    if args.dataset == 'imagenet':
        args.logger(f"\nLoading pretrained {args.model} for ImageNet from Torch model hub...")
        if args.model == 'resnet18':
            model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        elif args.model == 'resnet50':
            model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        elif args.model == 'resnet101':
            model = models.resnet101(weights='ResNet101_Weights.DEFAULT')            
        elif args.model == 'resnet152':
            model = models.resnet152(weights='ResNet152_Weights.DEFAULT')
        elif args.model == 'swin_t':
            model = models.swin_v2_t(weights='DEFAULT')

    elif args.dataset == 'cifar100':
        if args.model == 'resnet18':
            model = resnet.resnet18(num_classes=100)
    elif args.dataset == 'cifar10':
        if args.model == 'resnet18':
            model = resnet.resnet18(num_classes=10)
        elif args.model == 'vgg8':
            model = vgg.vgg8()

    # Try to load user trained model
    if args.dataset != 'imagenet':
        print(f"\nLoading pretrained model {saved_model}...")
        state_dict = torch.load(saved_model)

        # if model was saved after training with nn.DataParallel, remove 'module' from layer names
        for key in list(state_dict.keys()):
            if 'module' in key:
                state_dict[key.replace('module.', '')] = state_dict[key]
                del state_dict[key]

        model.load_state_dict(state_dict)


    model.to("cuda")

    print(model)

    # Create a list of all layers mapped to CIM
    layers = []

    for name, module in model.named_modules():
        if isinstance(module, macro.CIM):
            # Turn on input and weight quantization mode
            module._cim_args.quant_mode = 'iw'
            layers.append(name)
            module._cim_args.name = name

              
    # Pick which layers to quantize
    layer_quant = layers

    # Skip quantization for first and last layer
    # layer_quant = layers[1:-1]

    # Skip quantization for first layer
    layer_quant = layers[1:]

    print("Evaluating baseline model...\n")
    baseline_accuracy = evaluate(model, args, criterion, data_loader_test, num_batches=1, print_freq=1)

    # Step 3: Quantize model inputs and weights
    print('Quantizing inputs and weights...')

    # Collect histograms of inputs and weights
    print("Collecting histograms...\n")
    # layer_quant modified from layer_quant to layers in order to calculate amax for all layers
    collect_stats(model, layers, data_loader, args.gpu, quant_mode='iw', num_batches=2)
    strict = True
    if args.model == 'swin_t':
        strict = False
    print('\nComputing amax... \n')
    # layer_quant modified from layer_quant to layers in order to get the amax for all layers
    compute_amax(model, args, quant_mode='iw', layers=layers, method="percentile", percentile=99.9999, strict=strict)

    print("Finished quantizing inputs and weights.")
    print("Evaluating input and weight quantized model...\n")
    iw_quant_accuracy = evaluate(model, args, criterion, data_loader_test, num_batches=1, print_freq=1)

    for name, module in model.named_modules():
        if isinstance(module, macro.CIM):
            module._cim_args.quant_mode = 'adc'
        layer_name = name.split('._')[0]
        # disable integer calculation for unquantized layers
        if layer_name not in layer_quant:
            if isinstance(module, quant_nn.TensorQuantizer):
                module.disable()

    # Restore hardware mode
    args.hardware = hardware
    for name, module in model.named_modules():
        if isinstance(module, macro.CIM):
            module._cim_args.hardware = hardware
        
    return model

def evaluate(model, args, criterion, data_loader, num_batches, print_freq=100, log_suffix="", filename=None):

    device = torch.device(f"cuda:{args.gpu}")

    model.eval()

    start = time.time()
    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = 0
        test_count = 0
        for i, (images, labels) in tqdm(enumerate(data_loader), total=num_batches):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            test_count += 1

            print('Test Accuracy of the model on {} test images: {} %'.format((i+1) * args.batch_size, 100 * correct / total))
            print('Average Test loss: {}\n'.format(total_loss))
            
            if test_count == num_batches:
                total_loss /= test_count
                print("--------------------------------------------")
                print('evaluation time: {}'.format(time.time() - start))
                return 100 * correct / total

        total_loss /= test_count

        args.logger('Test Accuracy of the model on the 50000 test images: {} %'.format(100 * correct / total))
        args.logger('Average Test loss: {}\n'.format(total_loss))


        return 100 * correct / total


def collect_stats(model, layer_quant, data_loader, gpu, quant_mode='iw',  num_batches=2):
    """Feed data to the network and collect statistic"""

    for name, module in model.named_modules():
        if isinstance(module, macro.CIM):
            module._cim_args.calib = True

    # Enable calibrators
    for quant_name, module in model.named_modules():
        layer_name = quant_name.split('._')[0]
        if isinstance(module, quant_nn.TensorQuantizer):
            if layer_name in layer_quant:
                if quant_mode == 'iw' and 'adc' not in quant_name and module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                
                elif quant_mode == 'adc' and 'adc' in quant_name and module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
            else:
                module.disable()
   
   # can't use dataparallel here because histograms are collected using cpu
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(data_loader), total=num_batches):
            model(images.cuda())
            if i >= num_batches:
                break
                
    # Disable calibrators
    for quant_name, module in model.named_modules():
        layer_name = quant_name.split('._')[0]
        if isinstance(module, quant_nn.TensorQuantizer):
            if layer_name in layer_quant:
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()
            else:
                module.disable()

def compute_amax(model, args, layers, quant_mode='iw', **kwargs):
    # Load calib result
    for quant_name, module in model.named_modules():
        layer_name = quant_name.split('._')[0]
        if layer_name in layers:
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None :
                    if quant_mode == 'iw' and 'adc' not in quant_name:
                        if isinstance(module._calibrator, calib.MaxCalibrator):
                            module.load_calib_amax(strict=False)
                        else:
                            module.load_calib_amax(**kwargs)

                    elif quant_mode == 'adc' and 'adc' in quant_name:
                        if isinstance(module._calibrator, calib.MaxCalibrator):
                            module.load_calib_amax(strict=False)
                        else:
                            module.load_calib_amax(**kwargs)
        else:
            if isinstance(module, quant_nn.TensorQuantizer):
                module.disable() 

        if isinstance(module, macro.CIM):
            # turn off calibration mode
            module._cim_args.calib = False

    model.cuda()

def sample_conductance(model):
    for name, module in model.named_modules():
        if isinstance(module, macro.CIM):
            if hasattr(module, 'weight'):
                if (module.mem_sampling_seed.numel()!=0):
                    module.name = name
                    module.mem_sampling_seed = torch.randint(1, 1000, module.mem_sampling_seed.size())
                
                
            
                
            
