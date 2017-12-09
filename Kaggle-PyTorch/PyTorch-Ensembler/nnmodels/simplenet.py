import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import math

use_gpu = torch.cuda.is_available()


# class SimpleNet(nn.Module):
#     def __init__(self, num_classes=1, n_dim=3):
#         super(SimpleNet, self).__init__()
#         self.conv1 = nn.Conv2d(n_dim, 32, 3, stride=1)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
#
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
#         self.dense1 = nn.Linear(179776, out_features=512)
#         self.dense1_bn = nn.BatchNorm1d(512)
#         self.dense2 = nn.Linear(512, (num_classes))
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(F.dropout(F.max_pool2d(self.conv2(x), 2), 0.25))
#         x = F.relu(self.conv3(x))
#         x = F.relu(F.dropout(F.max_pool2d(self.conv4(x), 2), 0.25))
#         x = x.view(x.size(0), -1)
# #         print (x.data.shape)
#         x = F.relu(self.dense1_bn(self.dense1(x)))
#         x = x.view(x.size(0), -1)
# #         print (x.data.shape)
#         x = self.dense2(x)
#
#         return x

dropout = torch.nn.Dropout(p=0.30)
relu=torch.nn.LeakyReLU()
pool = nn.MaxPool2d(2, 2)


class ConvRes(nn.Module):
    def __init__(self, insize, outsize):
        super(ConvRes, self).__init__()
        drate = .3
        self.math = nn.Sequential(
            nn.BatchNorm2d(insize),
            nn.Dropout(drate),
            torch.nn.Conv2d(insize, outsize, kernel_size=2, padding=2),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.math(x)


class ConvCNN(nn.Module):
    def __init__(self, insize, outsize, kernel_size=7, padding=2, pool=2, avg=True):
        super(ConvCNN, self).__init__()
        self.avg = avg
        self.math = torch.nn.Sequential(
            torch.nn.Conv2d(insize, outsize, kernel_size=kernel_size, padding=padding),
            torch.nn.BatchNorm2d(outsize),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(pool, pool),
        )
        self.avgpool = torch.nn.AvgPool2d(pool, pool)

    def forward(self, x):
        x = self.math(x)
        if self.avg is True:
            x = self.avgpool(x)
        return x


class SimpleNet(nn.Module):
    def __init__(self,num_classes, n_dim):
        super(SimpleNet, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.cnn1 = ConvCNN(n_dim, 64, kernel_size=7, pool=4, avg=False)
        self.cnn2 = ConvCNN(64, 64, kernel_size=5, pool=2, avg=True)
        self.cnn3 = ConvCNN(64, 256, kernel_size=5, pool=2, avg=True)

        self.res1 = ConvRes(256, 64)

        self.features = nn.Sequential(
            self.cnn1, dropout,
            self.cnn2,
            self.cnn3,
            self.res1,
        )

        self.classifier = torch.nn.Sequential(
            nn.Linear(2304, (num_classes)),
        )

    #         self.sig=nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        #         print (x.data.shape)
        x = self.classifier(x)
        #         x = self.sig(x)
        return x

    #         return F.log_softmax(x)


def simpleXX_generic(num_classes, imgDim):
    # depth, num_classes = 1, widen_factor = 1, dropRate = 0.0
    model = SimpleNet(num_classes=num_classes, n_dim=imgDim)  # 56
    return model
