
import math

import torch.nn as nn
from torchvision.models import ResNet

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet18(num_classes):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet101(num_classes):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


REDUCTION=8

class SELayer(nn.Module):
    def __init__(self, channel, reduction=REDUCTION):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduction),
            # nn.ReLU(inplace=True),
            nn.PReLU(),
            nn.Linear(reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class IceSEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, reduction=REDUCTION):
        super(IceSEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1,
                                                  stride=1, bias=False),
                                        nn.BatchNorm2d(planes))

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class IceResNet(nn.Module):
    def __init__(self, block, n_size=1, num_classes=1, num_rgb=2, base=32):
        super(IceResNet, self).__init__()
        self.base = base
        self.num_classes = num_classes
        self.inplane = self.base  # 45 epochs
        # self.inplane = 16 # 57 epochs
        self.conv1 = nn.Conv2d(num_rgb, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, self.inplane, blocks=2 * n_size, stride=2)
        self.layer2 = self._make_layer(block, self.inplane * 2, blocks=2 * n_size, stride=2)
        self.layer3 = self._make_layer(block, self.inplane * 4, blocks=2 * n_size, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(int(8 * self.base), num_classes)
        nn.init.kaiming_normal(self.fc.weight)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride):

        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplane, planes, stride))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print (x.data.size())
        x = self.fc(x)

        if self.num_classes == 1:  # BCE Loss,
            x = self.sig(x)
        return x


def senet16_RGB_10_classes(num_classes=10, num_rgb=3):
    model = IceResNet(IceSEBasicBlock, 1, num_classes, num_rgb, 16)  # 56
    return model


def senet16_RG_1_classes(num_classes=1, num_rgb=2):
    model = IceResNet(IceSEBasicBlock, 1, num_classes, num_rgb, 16)  # 56
    return model


def senet32_RG_1_classes(num_classes=1, num_rgb=2):
    model = IceResNet(IceSEBasicBlock, 1, num_classes, num_rgb, 32)  # 56
    return model

def senet32_RGB_1_classes(num_classes=1, num_rgb=3):
    model = IceResNet(IceSEBasicBlock, 1, num_classes, num_rgb, 32)  # 56
    return model

def senet64_RG_1_classes(num_classes=1, num_rgb=2):
    model = IceResNet(IceSEBasicBlock, 1, num_classes, num_rgb, 64)  # 56
    return model


def senet128_RG_1_classes(num_classes=1, num_rgb=2):
    model = IceResNet(IceSEBasicBlock, 1, num_classes, num_rgb, 128)  # 56
    return model

def senetXX_generic(num_classes, num_rgb, base):
    model = IceResNet(IceSEBasicBlock, 1, num_classes, num_rgb, base)  # 56
    return model
