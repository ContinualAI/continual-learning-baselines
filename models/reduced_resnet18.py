import torch
from avalanche.models import MultiHeadClassifier, MultiTaskModule
from torch import nn, relu
from torch.nn.functional import avg_pool2d

"""
START: FROM GEM CODE https://github.com/facebookresearch/GradientEpisodicMemory/
CLASSIFIER REMOVED AND SUBSTITUTED WITH AVALANCHE MULTI-HEAD CLASSIFIER
"""


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        return out


"""
END: FROM GEM CODE
"""

class MultiHeadReducedResNet18(MultiTaskModule):
    """
    As from GEM paper, a smaller version of ResNet18, with three times less feature maps across all layers.
    It employs multi-head output layer.
    """

    def __init__(self, size_before_classifier=160):
        super().__init__()
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], 20)
        self.classifier = MultiHeadClassifier(size_before_classifier)

    def forward(self, x, task_labels):
        out = self.resnet(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out, task_labels)


class SingleHeadReducedResNet18(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], 20)
        self.classifier = nn.Linear(160, num_classes)

    def feature_extractor(self, x):
        out = self.resnet(x)
        return out.view(out.size(0), -1)

    def forward(self, x):
        out = self.feature_extractor(x)
        return self.classifier(out)


__all__ = ['MultiHeadReducedResNet18', 'SingleHeadReducedResNet18']
