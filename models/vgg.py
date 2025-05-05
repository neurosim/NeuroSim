import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, layers, num_classes):
        super(VGG, self).__init__()
        assert isinstance(layers, nn.Sequential), type(layers)
        self.layers = layers
        self.classifier = make_layers([('L', 8192, 1024),
                                       ('L', 1024, num_classes)])

        # print(self.layers)
        # print(self.classifier)

    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=v[1], stride=v[2])]
        if v[0] == 'C':
            out_channels = v[1]
            if v[3] == 'same':
                padding = v[2]//2
            else:
                padding = 0
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=v[2], padding=padding)
            non_linearity_activation =  nn.ReLU()
            layers += [conv2d, non_linearity_activation]
            in_channels = out_channels
        if v[0] == 'L':
            linear = nn.Linear(in_features=v[1], out_features=v[2])
            if i < len(cfg)-1:
                non_linearity_activation =  nn.ReLU()
                layers += [linear, non_linearity_activation]
            else:
                layers += [linear]
    return nn.Sequential(*layers)



cfg_list = {
    'vgg8': [('C', 128, 3, 'same', 2.0),
                ('C', 128, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 256, 3, 'same', 16.0),
                ('C', 256, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 32.0),
                ('M', 2, 2)]
}

def vgg8(pretrained=None):
    cfg = cfg_list['vgg8']
    layers = make_layers(cfg)
    model = VGG(layers, num_classes=10)
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        for key in list(state_dict.keys()):
            if 'module' in key:
                state_dict[key.replace('module.', '')] = state_dict[key]
                del state_dict[key]

        # model.load_state_dict(torch.load(pretrained))
        model.load_state_dict(state_dict)

    return model


