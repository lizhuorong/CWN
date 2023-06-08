import torch
import torch.nn as nn
import torch.nn.functional
from settings import Config


class MyWeightedLoss(torch.nn.Module):
    def __init__(self):
        super(MyWeightedLoss, self).__init__()
        self.weight = torch.zeros([Config.CLASS_NUM, Config.CLASS_NUM]).to(Config.DEVICE)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.alpha = torch.tensor([0.7, 0.3, 0.6, 0.3, 0.7]).to(Config.DEVICE)
        self.gamma = 5.0

        # weightA
        # for i in range(Config.CLASS_NUM):
        #     for j in range(Config.CLASS_NUM):
        #         self.weight[i, j] = 1 - (i - j) ** 2 / (Config.CLASS_NUM - 1)**2

        # # weightB
        # for i in range(Config.CLASS_NUM):
        #     for j in range(Config.CLASS_NUM):
        #         self.weight[i, j] = ((Config.CLASS_NUM - 1) - abs(i - j)) / (Config.CLASS_NUM - 1)

        # # weightC
        for i in range(Config.CLASS_NUM):
            for j in range(Config.CLASS_NUM):
                self.weight[i, j] = (((Config.CLASS_NUM - 1) - abs(i - j)) / (Config.CLASS_NUM - 1)) ** 2

    def forward(self, output, target):
        x = self.log_softmax(output)
        y = self.weight[target]
        return -torch.mean(torch.sum(torch.mul(x, y), dim=1))
