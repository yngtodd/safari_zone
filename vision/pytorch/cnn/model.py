import torch
import torch.nn as nn


# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__(kernel_size1=5, padding1=2, pool_kernel1=2,
                                  kernel_size2=5, padding2=2, pool_kernel2=2)
        self.kernel_size1
        self.padding1
        self.pool_kernel1
        self.kernel_size2
        self.padding2
        self.pool_kernel2

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=kernel_size1, padding=padding1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=kernel_size2, padding=padding2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
