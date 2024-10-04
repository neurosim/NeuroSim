from utee import misc
print = misc.logger.info
import torch.nn as nn
from modules.quantization_cpu_np_infer import QConv2d,  QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import torch
import math
name=0

class DenseNet(nn.Module):
    def __init__(self, args, logger, depth, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        nBlocks = (depth-4) // 3 // 2
        
        num_planes = 2*growth_rate
        self.conv1 = make_layers([('C', 3, num_planes, 3, 'same', 1)], args, logger)

        self.dense1 = self._make_dense(args, logger, num_planes, nBlocks, growth_rate)
        num_planes += nBlocks*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(args, logger, num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense(args, logger, num_planes, nBlocks, growth_rate)
        num_planes += nBlocks*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(args, logger, num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense(args, logger, num_planes, nBlocks, growth_rate)
        num_planes += nBlocks*growth_rate

        self.avgpool = nn.AvgPool2d(8)
        self.classifier = make_layers([('L', num_planes, num_classes)], args, logger)
    
        for m in self.modules():
            if isinstance(m, QConv2d) or isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
    def _make_dense(self, args, logger, in_planes, nblock, growth_rate):
        layers = []
        for i in range(nblock):
            layers.append(Bottleneck(args, logger, in_planes, growth_rate))
            in_planes += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, args, logger, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.conv = make_layers([('C', in_planes, 4*growth_rate, 1, 'same', 1),
                                  ('C', 4*growth_rate, growth_rate, 3, 'same', 1)], 
                                  args, logger)

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, args, logger, in_planes, out_planes):
        super(Transition, self).__init__()
        self.conv = make_layers([('C', in_planes, out_planes, 1, 'same', 1)], args, logger)
        self.avgpool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.conv(x)
        out = self.avgpool(out)
        return out


def make_layers(cfg, args, logger):
    global name
    layers = []
    for i, v in enumerate(cfg):
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=v[1], stride=v[2])]
        if v[0] == 'C':
            in_channels = v[1]
            out_channels = v[2]
            if v[4] == 'same':
                padding = v[3]//2
            else:
                padding = 0
            if args.mode == "WAGE":
                conv2d = QConv2d(in_channels, out_channels, kernel_size=v[3], stride=v[5], padding=padding,
                                 logger=logger,wl_input = args.wl_activate,wl_activate=args.wl_activate,
                                 wl_error=args.wl_error,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                                 subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                                 name = 'Conv'+str(name)+'_', model = args.model, parallelRead=args.parallelRead)
            elif args.mode == "FP":
                conv2d = FConv2d(in_channels, out_channels, kernel_size=v[3], stride=v[5], padding=padding,
                                 logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                                 subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                                 name = 'Conv'+str(name)+'_' )
            name += 1
            batchnorm = nn.BatchNorm2d(out_channels)
            non_linearity_activation =  nn.ReLU()
            layers += [conv2d, batchnorm, non_linearity_activation]
            in_channels = out_channels
        if v[0] == 'L':
            if args.mode == "WAGE":
                linear = QLinear(in_features=v[1], out_features=v[2], 
                                logger=logger, wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                                wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                                subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target, 
                                name='FC'+str(i)+'_', model = args.model, parallelRead=args.parallelRead)
            elif args.mode == "FP":
                linear = FLinear(in_features=v[1], out_features=v[2],
                                 logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                                 subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                                 name='FC'+str(i)+'_')
            layers += [linear]
    return nn.Sequential(*layers)
    
    
def densenet40(args, logger, pretrained=None):
    model = DenseNet(args, logger, depth=40, growth_rate=12, reduction=0.5, num_classes=10)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model
    
    