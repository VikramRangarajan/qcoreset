import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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

    def noisy_shortcut(self, shortcut, x, noise_std):
        out = x
        if isinstance(shortcut, nn.Sequential):
            for layer in shortcut:
                if isinstance(layer, nn.Conv2d):
                    noisy_weight = (
                        layer.weight + torch.randn_like(layer.weight) * noise_std
                    )
                    out = F.conv2d(
                        out,
                        noisy_weight,
                        bias=None,
                        stride=layer.stride,
                        padding=layer.padding,
                    )
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
        else:
            out = shortcut(out)
        return out

    def forward(self, x):
        # Apply noise to conv1 if USE_NOISE is enabled
        if config.USE_NOISE:
            noisy_weight = (
                self.conv1.weight + torch.randn_like(self.conv1.weight) * self.noise_std
            )
            out = F.conv2d(
                x,
                noisy_weight,
                None,
                stride=self.conv1.stride,
                padding=self.conv1.padding,
            )
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

        # Apply noise to conv2
        if config.USE_NOISE:
            noisy_weight = (
                self.conv2.weight + torch.randn_like(self.conv2.weight) * self.noise_std
            )
            out = F.conv2d(
                out,
                noisy_weight,
                None,
                stride=self.conv2.stride,
                padding=self.conv2.padding,
            )
            out = F.batch_norm(
                out,
                self.bn2.running_mean,
                self.bn2.running_var,
                self.bn2.weight + torch.randn_like(self.bn2.weight) * self.noise_std,
                self.bn2.bias + torch.randn_like(self.bn2.bias) * self.noise_std,
                self.bn2.training,
            )
        else:
            out = self.bn2(self.conv2(out))

        if config.USE_NOISE:
            shortcut = self.noisy_shortcut(self.shortcut, x, self.noise_std)
        else:
            shortcut = self.shortcut(x)

        out += shortcut
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, std=0):
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
        self.noise_std = std

    def noisy_shortcut(self, shortcut, x, noise_std):
        out = x
        if isinstance(shortcut, nn.Sequential):
            for layer in shortcut:
                if isinstance(layer, nn.Conv2d):
                    noisy_weight = (
                        layer.weight + torch.randn_like(layer.weight) * noise_std
                    )
                    out = F.conv2d(
                        out,
                        noisy_weight,
                        bias=None,
                        stride=layer.stride,
                        padding=layer.padding,
                    )
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
        else:
            out = shortcut(out)
        return out

    def forward(self, x):
        if config.USE_NOISE:
            noisy_weight = (
                self.conv1.weight + torch.randn_like(self.conv1.weight) * self.noise_std
            )
            out = F.conv2d(
                x,
                noisy_weight,
                None,
                stride=self.conv1.stride,
                padding=self.conv1.padding,
            )
            out = F.batch_norm(
                out,
                self.bn1.running_mean,
                self.bn1.running_var,
                self.bn1.weight + torch.randn_like(self.bn1.weight) * self.noise_std,
                self.bn1.bias + torch.randn_like(self.bn1.bias) * self.noise_std,
                self.bn1.training,
            )
            out = F.relu(out)
            noisy_weight = (
                self.conv2.weight + torch.randn_like(self.conv2.weight) * self.noise_std
            )
            out = F.conv2d(
                out,
                noisy_weight,
                None,
                stride=self.conv2.stride,
                padding=self.conv2.padding,
            )
            out = F.batch_norm(
                out,
                self.bn2.running_mean,
                self.bn2.running_var,
                self.bn2.weight + torch.randn_like(self.bn2.weight) * self.noise_std,
                self.bn2.bias + torch.randn_like(self.bn2.bias) * self.noise_std,
                self.bn2.training,
            )
            out = F.relu(out)
            noisy_weight = (
                self.conv3.weight + torch.randn_like(self.conv3.weight) * self.noise_std
            )
            out = F.conv2d(
                out,
                noisy_weight,
                None,
                stride=self.conv3.stride,
                padding=self.conv3.padding,
            )
            out = F.batch_norm(
                out,
                self.bn3.running_mean,
                self.bn3.running_var,
                self.bn3.weight + torch.randn_like(self.bn3.weight) * self.noise_std,
                self.bn3.bias + torch.randn_like(self.bn3.bias) * self.noise_std,
                self.bn3.training,
            )
            shortcut = self.noisy_shortcut(self.shortcut, x, self.noise_std)
            out += shortcut
            out = F.relu(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        std=0,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.noise_std = std
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                std=self.noise_std,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        if config.USE_NOISE:
            noisy_weight = (
                self.conv1.weight + torch.randn_like(self.conv1.weight) * self.noise_std
            )
            out = F.conv2d(
                x,
                noisy_weight,
                None,
                stride=self.conv1.stride,
                padding=self.conv1.padding,
            )
            out = F.batch_norm(
                out,
                self.bn1.running_mean,
                self.bn1.running_var,
                self.bn1.weight + torch.randn_like(self.bn1.weight) * self.noise_std,
                self.bn1.bias + torch.randn_like(self.bn1.bias) * self.noise_std,
                self.bn1.training,
            )
            out = F.relu(out)
            out = self.maxpool(out)
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        if config.USE_NOISE:
            noisy_weight = (
                self.linear.weight
                + torch.randn_like(self.linear.weight) * self.noise_std
            )
            noisy_bias = (
                self.linear.bias + torch.randn_like(self.linear.bias) * self.noise_std
                if self.linear.bias is not None
                else None
            )
            out = F.linear(out, noisy_weight, noisy_bias)
        else:
            out = self.linear(out)
        return out

    def forward(self, x):
        return self._forward_impl(x)


def ResNet50(num_classes=200, std=0):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, std=std)
