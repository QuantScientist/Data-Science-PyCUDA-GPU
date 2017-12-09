'''VGG11/13/16/19 in Pytorch.'''
import torch.nn as nn

__all__ = ['vggnetXX_generic']

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, num_rgb):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], num_rgb)
        self.classifier = nn.Linear(25088, num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # print (out.data.size())
        out = self.classifier(out)
        # out = self.sig(out)
        return out

    def _make_layers(self, cfg, num_rgb):
        layers = []
        in_channels = num_rgb
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


# net = VGG('VGG11')
# x = torch.randn(2,3,32,32)
# print(net(Variable(x)).size())

def vggnetXX_generic(num_classes, num_rgb):
    model = VGG('VGG19', num_classes, num_rgb)  # 56
    return model
