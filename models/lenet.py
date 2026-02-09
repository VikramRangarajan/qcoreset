import torch.nn as nn
import torch.nn.functional as F
import torch
from . import config


class lenet(nn.Module):
    def __init__(self, num_classes, std):
        super(lenet, self).__init__()
        self.embDim = 84
        self.num_classes = num_classes
        self.noise_std = std
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        if config.USE_NOISE:
            noisy_weight = (
                self.conv1.weight + torch.randn_like(self.conv1.weight) * self.noise_std
            )
            noisy_bias = (
                self.conv1.bias + torch.randn_like(self.conv1.bias) * self.noise_std
            )
            out = F.relu(
                F.conv2d(
                    x,
                    noisy_weight,
                    noisy_bias,
                    stride=self.conv1.stride,
                    padding=self.conv1.padding,
                )
            )
            out = F.max_pool2d(out, 2)

            noisy_weight = (
                self.conv2.weight + torch.randn_like(self.conv2.weight) * self.noise_std
            )
            noisy_bias = (
                self.conv2.bias + torch.randn_like(self.conv2.bias) * self.noise_std
            )
            out = F.relu(
                F.conv2d(
                    out,
                    noisy_weight,
                    noisy_bias,
                    stride=self.conv2.stride,
                    padding=self.conv2.padding,
                )
            )
            out = F.max_pool2d(out, 2)

            out = out.view(out.size(0), -1)
            noisy_weight = (
                self.fc1.weight + torch.randn_like(self.fc1.weight) * self.noise_std
            )
            noisy_bias = (
                self.fc1.bias + torch.randn_like(self.fc1.bias) * self.noise_std
                if self.fc1.bias is not None
                else None
            )
            out = F.relu(F.linear(out, noisy_weight, noisy_bias))

            noisy_weight = (
                self.fc2.weight + torch.randn_like(self.fc2.weight) * self.noise_std
            )
            noisy_bias = (
                self.fc2.bias + torch.randn_like(self.fc2.bias) * self.noise_std
                if self.fc2.bias is not None
                else None
            )
            out = F.relu(F.linear(out, noisy_weight, noisy_bias))

            noisy_weight = (
                self.fc3.weight + torch.randn_like(self.fc3.weight) * self.noise_std
            )
            noisy_bias = (
                self.fc3.bias + torch.randn_like(self.fc3.bias) * self.noise_std
                if self.fc3.bias is not None
                else None
            )
            out = F.relu(F.linear(out, noisy_weight, noisy_bias))

        else:
            out = F.relu(self.conv1(x))
            out = F.max_pool2d(out, 2)
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            e = F.relu(self.fc2(out))
            out = self.fc3(e)

        return out

    def get_embedding_dim(self):
        return self.embDim


def LeNet(num_classes=10, std=0):
    return lenet(num_classes=num_classes, std=std)
