"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, std=0.01):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.noise_std = std
        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def noisy_shortcut(self, shortcut, x, noise_std):
        out = x
        if isinstance(shortcut, nn.Sequential):
            for layer in shortcut:
                if isinstance(layer, nn.Conv2d):
                    out = layer(out)
                elif isinstance(layer, nn.BatchNorm2d):
                    noisy_weight = (
                        layer.weight + torch.randn_like(layer.weight) * noise_std
                    )
                    noisy_bias = layer.bias + torch.randn_like(layer.bias) * noise_std
                    out = F.batch_norm(
                        out,
                        layer.running_mean,
                        layer.running_var,
                        noisy_weight,
                        noisy_bias,
                        layer.training,
                    )
        return out

    def forward(self, x):
        # Apply noise to conv1 if USE_NOISE is enabled
        if config.USE_NOISE:
            out = self.conv1(x)
            out = F.batch_norm(
                out,
                self.bn1.running_mean,
                self.bn1.running_var,
                self.bn1.weight + torch.randn_like(self.bn1.weight) * self.noise_std,
                self.bn1.bias + torch.randn_like(self.bn1.bias) * self.noise_std,
                self.bn1.training,
            )
            out = F.relu(out)

            out = self.conv2(out)
            out = F.batch_norm(
                out,
                self.bn2.running_mean,
                self.bn2.running_var,
                self.bn2.weight + torch.randn_like(self.bn2.weight) * self.noise_std,
                self.bn2.bias + torch.randn_like(self.bn2.bias) * self.noise_std,
                self.bn2.training,
            )
            if self.shortcut is None:
                out += x
            else:
                out += self.noisy_shortcut(self.shortcut, x, self.noise_std)
            out = F.relu(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.shortcut is None:
                out += x
            else:
                out += self.shortcut(x)
            out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, std=0.01):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.noise_std = std
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.hold = [self.layer1, self.layer2, self.layer3, self.layer4]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, std=self.noise_std))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if config.USE_NOISE:
            out = self.conv1(x)
            out = F.batch_norm(
                out,
                self.bn1.running_mean,
                self.bn1.running_var,
                self.bn1.weight + torch.randn_like(self.bn1.weight) * self.noise_std,
                self.bn1.bias + torch.randn_like(self.bn1.bias) * self.noise_std,
                self.bn1.training,
            )
            out = F.relu(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        last_input = out.view(out.size(0), -1)
        out = self.linear(last_input)

        return out

    def update_noise(self, noise):
        self.noise_std = noise
        for layer in self.hold:
            for block in layer:
                block.noise_std = noise


def ResNet18_noise_bn(num_classes=10, std=0):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, std=std)
