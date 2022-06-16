import torch.nn as nn
from avalanche.models.dynamic_modules import MultiTaskModule,\
    MultiHeadClassifier


####################
#     CIFAR-100
####################

class ConvCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvCIFAR, self).__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 160, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 160, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 160, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # Linear layers
        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(16*160, 320)
        self.linear2 = nn.Linear(320, 320)
        # Classifier
        self.classifier = nn.Linear(320, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 2560)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))

        x = self.classifier(x)

        return x


class MTConvCIFAR(ConvCIFAR, MultiTaskModule):
    def __init__(self):
        super(MTConvCIFAR, self).__init__()
        # Classifier
        self.classifier = MultiHeadClassifier(320)

    def forward(self, x, task_labels):
        x = self.conv_layers(x)
        x = x.view(-1, 16*160)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.classifier(x, task_labels)

        return x


####################
#   TinyImageNet
####################

class ConvTinyImageNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvTinyImageNet, self).__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 160, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 160, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 160, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 160, kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # linear layers
        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(16*160, 640)
        self.linear2 = nn.Linear(640, 640)
        # classifier
        self.classifier = nn.Linear(640, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 16*160)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.classifier(x)

        return x


class MTConvTinyImageNet(ConvTinyImageNet, MultiTaskModule):
    def __init__(self):
        super(MTConvTinyImageNet, self).__init__()
        # Classifier
        self.classifier = MultiHeadClassifier(640)

    def forward(self, x, task_labels):
        x = self.conv_layers(x)
        x = x.view(-1, 16*160)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.classifier(x, task_labels)

        return x
