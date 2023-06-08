import torch
import torch.nn as nn
from torch import Tensor
import torchvision
from collections import OrderedDict
from unet import UNet
from EfficientNetV2 import efficientnetv2_s


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv_block(in_planes: int):
    layers = []
    layers.append(conv1x1(in_planes, in_planes))
    layers.append(nn.BatchNorm2d(in_planes))
    layers.append(nn.ReLU(inplace=True))
    layers.append(conv1x1(in_planes, in_planes))
    layers.append(nn.BatchNorm2d(in_planes))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class CWN(nn.Module):
    def __init__(
            self,
            baseline: nn.Module,
            in_planes: int,
            classes: int,
            k: list = []
    ) -> None:
        super(CWN, self).__init__()
        self.baseline = baseline
        self.in_planes = in_planes
        self.classes = classes
        self.k = k

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_block1 = conv_block(self.in_planes)
        self.conv_block2 = conv_block(1)

        self.conv1 = conv1x1(self.in_planes, sum(self.k))
        self.bn1 = nn.BatchNorm2d(sum(self.k))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.conv_block3 = conv_block(self.classes)
        self.conv_block4 = conv_block(1)

        self.fc1 = nn.Linear(self.in_planes, self.in_planes // 2)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.in_planes // 2, self.classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.baseline(x)

        # chanel
        out1 = self.global_avgpool(x)
        out1 = self.conv_block1(out1)
        C_A = torch.mul(out1, x)

        # sapce
        out2 = torch.mean(x, dim=-3, keepdim=True)
        out2 = self.conv_block2(out2)
        S_A = torch.mul(out2, x)

        # category
        out3 = self.conv1(x)
        out3 = self.bn1(out3)
        out3 = self.relu(out3)
        out3 = self.dropout(out3)

        out3 = list(torch.split(out3, self.k, 1))
        for i, item in enumerate(out3):
            out3[i] = torch.mean(out3[i], dim=1, keepdim=False)
        S = torch.stack(out3, 1)

        out3 = self.global_maxpool(S)
        out3 = self.conv_block3(out3)
        T_A1 = torch.mul(S, out3)

        out3 = torch.mean(T_A1, dim=1, keepdim=True)
        out3 = self.conv_block4(out3)
        T_A2 = torch.mul(x, out3)

        out = (C_A + S_A + T_A2)/3
        out = self.global_avgpool(out)
        out = out.flatten(1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


def cwn(model_name='efficientnet', num_classes=5, K=[]):
    if model_name == 'densenet121':
        baseline = torchvision.models.densenet121(pretrained=False)
        state_dict = torch.load("BaseLine/densenet121-a639ec97.pth")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'denseblock' in k:
                param = k.split(".")
                k = ".".join(param[:-3] + [param[-3] + param[-2]] + [param[-1]])
            new_state_dict[k] = v
        baseline.load_state_dict(new_state_dict)
        baseline = nn.Sequential(*list(baseline.children())[:-1])
        inplace = 1024
    elif model_name == "unet":
        baseline = UNet(3, 2)
        baseline.load_state_dict(torch.load('BaseLine/unet_carvana_scale0.5_epoch2.pth'))
        inplace = 2
    elif model_name == "resnet50":
        baseline = torchvision.models.resnet50(pretrained=False)
        baseline.load_state_dict(torch.load('BaseLine/resnet50-0676ba61.pth'))
        baseline = nn.Sequential(*list(baseline.children())[:-2])
        inplace = 2048
    else:
        baseline = efficientnetv2_s()
        baseline.load_state_dict(torch.load('BaseLine/pre_efficientnetv2-s.pth'))
        baseline = nn.Sequential(*list(baseline.children())[:-1])
        inplace = 256

    return CWN(baseline, inplace, num_classes, K)
