'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['densnetXX_generic', 'DenseNet169', 'DenseNet121']


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

        self.mp = torch.nn.MaxPool2d(1, 1)
        # self.avgpool = torch.nn.AvgPool2d(2,2)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))

        out = self.conv2(F.relu(self.bn2(out)))
        # out = self.mp(out)
        # out = self.avgpool(out)

        # print (x.data.shape)

        out = torch.cat([out, x], 1)
        out = self.mp(out)
        # out = self.avgpool(out)
        # print(out.data.shape)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=1, n_dim=3):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(n_dim, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)

        self.linear = nn.Linear(448, num_classes)
        self.sig = nn.Sigmoid()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        # print (out.data.shape)
        out = self.linear(out)
        out = self.sig(out)

        return out


def densnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return DenseNet(**kwargs)


# block, nblocks, growth_rate=12, reduction=0.5, num_classes=1, n_dim=3
def densnetXX_generic(num_classes, n_dim):
    model = DenseNet(Bottleneck, [6, 12, 24, 12], growth_rate=4, num_classes=num_classes, n_dim=n_dim)  # 56
    return model


def DenseNet121():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12)


def DenseNet169():
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=16)
