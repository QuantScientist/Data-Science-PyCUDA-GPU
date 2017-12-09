from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self, sign=1):
        super(BCELoss, self).__init__()
        self.sign = sign
        self.main = nn.BCELoss()

    def forward(self, input, target):
        output = self.main(input, target)
        output = torch.mul(output, self.sign)
        return output

    def cuda(self, device_id=None):
        super(BCELoss, self).cuda(device_id)
        self.main.cuda()
