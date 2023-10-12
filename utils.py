import torch
from tools.logger import *


def set_device(hps):
    if hps.cuda and hps.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("[INFO] Use cuda")
    else:
        device = torch.device("cpu")
        logger.info("[INFO] Use CPU")
    hps.device = device
    return hps


import torch
from torch import nn


class Power2Loss(nn.Module):
    def __init__(self):
        super(Power2Loss, self).__init__()
        self.L1Loss = nn.L1Loss(reduction='none')

    def forward(self, output, target):
        return self.L1Loss(output ** 2, target ** 2) * 100


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, output, target):
        return self.loss(output, target) * (target + 1) ** 2


class RegressionCustomLoss(nn.Module):
    def __init__(self):
        super(RegressionCustomLoss, self).__init__()
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, output, target):
        return self.loss(output, target) * (1 + target + output) ** 2


if __name__ == '__main__':
    input = torch.Tensor([0.9, 0.5, 0.2, 0.6])
    out = torch.Tensor([1, 0.6, 0.7, 0.3])
    criterion = RegressionCustomLoss()
    print(criterion(input, out))
