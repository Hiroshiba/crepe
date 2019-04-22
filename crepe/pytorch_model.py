import torch
import torch.nn as nn
import torch.nn.functional as F


class PytorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1024, kernel_size=(512, 1), stride=(4, 1))
        self.conv1_BN = nn.BatchNorm2d(num_features=1024, eps=0.0010000000474974513, momentum=0.0)
        self.conv2 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=(64, 1), stride=(1, 1))
        self.conv2_BN = nn.BatchNorm2d(num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(64, 1), stride=(1, 1))
        self.conv3_BN = nn.BatchNorm2d(num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(64, 1), stride=(1, 1))
        self.conv4_BN = nn.BatchNorm2d(num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(64, 1), stride=(1, 1))
        self.conv5_BN = nn.BatchNorm2d(num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(64, 1), stride=(1, 1))
        self.conv6_BN = nn.BatchNorm2d(num_features=512, eps=0.0010000000474974513, momentum=0.0)
        self.classifier = nn.Linear(in_features=2048, out_features=360)

    def forward(self, x):
        x = torch.reshape(x, shape=(-1, 1, 1024, 1))

        x = F.pad(x, (0, 0, 254, 254))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv1_BN(x)
        x = F.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1), padding=0, ceil_mode=False)
        x = F.dropout(input=x, p=0.25, training=self.training, inplace=True)

        x = F.pad(x, (0, 0, 31, 32))
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv2_BN(x)
        x = F.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1), padding=0, ceil_mode=False)
        x = F.dropout(input=x, p=0.25, training=self.training, inplace=True)

        x = F.pad(x, (0, 0, 31, 32))
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv3_BN(x)
        x = F.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1), padding=0, ceil_mode=False)
        x = F.dropout(input=x, p=0.25, training=self.training, inplace=True)

        x = F.pad(x, (0, 0, 31, 32))
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv4_BN(x)
        x = F.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1), padding=0, ceil_mode=False)
        x = F.dropout(input=x, p=0.25, training=self.training, inplace=True)

        x = F.pad(x, (0, 0, 31, 32))
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv5_BN(x)
        x = F.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1), padding=0, ceil_mode=False)
        x = F.dropout(input=x, p=0.25, training=self.training, inplace=True)

        x = F.pad(x, (0, 0, 31, 32))
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv6_BN(x)
        x = F.max_pool2d(x, kernel_size=(2, 1), stride=(2, 1), padding=0, ceil_mode=False)
        x = F.dropout(input=x, p=0.25, training=self.training, inplace=True)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.sigmoid(x)
        return x
