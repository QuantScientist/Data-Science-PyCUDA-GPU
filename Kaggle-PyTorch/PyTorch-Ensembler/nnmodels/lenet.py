'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['lenetXX_generic']


class LeNet(nn.Module):
    def __init__(self, num_classes=12, num_rgb=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_rgb, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(44944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # print(out.data.size())
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = out.view(out.size(0), -1)
        # print(out.data.size())
        out = self.fc3(out)
        # out = self.sig(out)
        return out


def lenetXX_generic(num_classes, num_rgb):
    model = LeNet(num_classes, num_rgb)  # 56
    return model
