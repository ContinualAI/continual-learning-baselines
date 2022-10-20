import torch
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
