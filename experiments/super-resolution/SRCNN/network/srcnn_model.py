import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class SRCNN2D(nn.Module):
    def __init__(self, num_channels, name='srcnn2d'):
        super(SRCNN2D, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        
    def forward(self,x):
        h = self.conv1(x)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.conv3(h)

        return h


class MODIF_SRCNN2D(nn.Module):
    def __init__(self, num_channels, name='modified_srcnn2d'):
        super(MODIF_SRCNN2D, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        
    def forward(self,x):
        h = self.conv1(x)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.conv3(h)
        h = self.relu3(h)
        h = self.conv4(h)
        h = self.relu4(h)
        h = self.conv5(h)
        h = self.relu5(h)
        h = self.conv6(h)
        h = self.relu6(h)
        h = self.conv7(h)

        return h
