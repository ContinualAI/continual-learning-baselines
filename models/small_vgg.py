"""
Small VGG net adapted from https://github.com/Mattdl/CLsurvey/
"""

import torch.nn as nn
import torch
import torchvision
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence

from avalanche.models import MultiTaskModule

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
        x = self.avgpool(x)
        return x


class MultiHeadVGGClassifier(MultiTaskModule):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.classifiers = torch.nn.ModuleDict()
        first_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, self.n_classes))
        self.classifiers['0'] = first_head

    def adaptation(self, experience):
        super().adaptation(experience)
        task_labels = experience.dataset.targets_task_labels
        if isinstance(task_labels, ConstantSequence):
            # task label is unique. Don't check duplicates.
            task_labels = [task_labels[0]]

        for tid in set(task_labels):
            tid = str(tid)  # need str keys
            if tid not in self.classifiers:
                new_head = nn.Sequential(
                    nn.Linear(self.in_features, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, self.n_classes))
                self.classifiers[tid] = new_head

    def forward_single_task(self, x, task_label):
        """ compute the output given the input `x`. This module uses the task
        label to activate the correct head.

        :param x:
        :param task_label:
        :return:
        """
        return self.classifiers[str(task_label)](x)


class MultiHeadVGGSmall(MultiTaskModule):
    def __init__(self, n_classes=20):
        super().__init__()
        self.vgg = VGGSmall()
        self.classifier = MultiHeadVGGClassifier(in_features=2048,
                                                 n_classes=n_classes)

    def forward(self, x, task_labels):
        x = self.vgg(x)
        x = torch.flatten(x, 1)
        return self.classifier(x, task_labels)
