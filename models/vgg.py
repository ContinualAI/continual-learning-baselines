import torch
from torch import nn
import torchvision
from avalanche.models import MultiTaskModule, MultiHeadClassifier


class MultiHeadVGG(MultiTaskModule):
    def __init__(self, n_classes=20):
        super().__init__()
        self.vgg = torchvision.models.vgg11()
        self.classifier = MultiHeadClassifier(in_features=1000, initial_out_features=n_classes)

    def forward(self, x, task_labels):
        x = self.vgg(x)
        x = torch.flatten(x, 1)
        return self.classifier(x, task_labels)


"""
Small VGG net adapted from https://github.com/Mattdl/CLsurvey/
"""

cfg = [64, 'M', 64, 'M', 64, 64, 'M', 128, 128, 'M']
conv_kernel_size = 3
img_input_channels = 3


class VGGSmall(torchvision.models.VGG):
    """
    Creates VGG feature extractor from config and custom classifier.
    """

    def __init__(self):

        in_channels = img_input_channels
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=conv_kernel_size, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        super(VGGSmall, self).__init__(nn.Sequential(*layers), init_weights=True)

        if hasattr(self, 'avgpool'):  # Compat Pytorch>1.0.0
            self.avgpool = torch.nn.Identity()

        del self.classifier

    def forward(self, x):
        x = self.features(x)
        return x


class MultiHeadVGGSmall(MultiTaskModule):
    def __init__(self, n_classes=200, hidden_size=128):
        super().__init__()
        self.vgg = VGGSmall()
        self.feedforward = nn.Sequential(
            nn.Linear(128*4*4, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
        )
        self.classifier = MultiHeadClassifier(in_features=128,
                                              initial_out_features=n_classes)

    def forward(self, x, task_labels):
        x = self.vgg(x)
        x = torch.flatten(x, 1)
        x = self.feedforward(x)
        return self.classifier(x, task_labels)


class SingleHeadVGGSmall(nn.Module):
    def __init__(self, n_classes=200, hidden_size=128):
        super().__init__()
        self.vgg = VGGSmall()
        self.feedforward = nn.Sequential(
            nn.Linear(128 * 4 * 4, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
        )
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = self.vgg(x)
        x = torch.flatten(x, 1)
        x = self.feedforward(x)
        return self.classifier(x)
