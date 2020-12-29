# encoding=utf-8
"""
    Created on 10:41 2019/07/07 
    @author: Avijoy Chakma
"""
import torch.nn as nn
import torch.nn.functional as F
import torch


class AccConvolution(nn.Module):
    def __init__(self):
        super(AccConvolution, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 9))
        self.selu1 = nn.SELU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices = True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 9))
        self.selu2 = nn.SELU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=2, return_indices = True)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 9))
        self.selu3 = nn.SELU()

    def forward(self, x):
        x = self.selu1(self.conv1(x))
        shape1 = x.shape 
        x, indices1 = self.maxpool1(x)
        x = self.selu2(self.conv2(x))
        shape2 = x.shape
        x, indices2 = self.maxpool2(x)
        x = self.selu3(self.conv3(x))
        
        return x, indices1, indices2, shape1, shape2
    
#         print("First conv: "+ str(x.shape))
#         print("First max pool: "+ str(x.shape))
#         print("Second conv: "+ str(x.shape))
#         print("Second max pool: "+ str(x.shape))
#         print("Third conv: "+ str(x.shape))
    