# encoding=utf-8
"""
    Created on 10:41 2019/07/07 
    @author: Avijoy Chakma
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
# Input: 64,3,1,64


class AccExtractor(nn.Module):
    def __init__(self):
        super(AccExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 9)),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 9)),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 9)),
            nn.SELU()
#             nn.MaxPool2d(kernel_size=(1, 2), stride=2),
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 9)),
#             nn.SELU()
        )


    def forward(self, x):
        res = self.features(x)
        return res
    
# class AccExtractor256(nn.Module):
#     def __init__(self):
#         super(AccExtractor256, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 9)),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.MaxPool2d(kernel_size=(1, 2), stride=2)
#         )
        
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 9)),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.MaxPool2d(kernel_size=(1, 2), stride=2)
#         )

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         return out
    
    
class SelectiveSequential(nn.Module):
    def __init__(self, to_select, modules_dict):
       
        super(SelectiveSequential, self).__init__()
        for key, module in modules_dict.items():
            self.add_module(key, module)
        self._to_select = to_select
    
    def forward(self, x):
        list = []
        for name, module in self._modules.items():
            x = module(x)
            if name in self._to_select:
                list.append(x)
            global output
            output = x
        return output, list
    

class AccExtractorOld(nn.Module):
    def __init__(self):
        
        super(AccExtractor, self).__init__()
        self.features = SelectiveSequential(
#             ['conv1', 'pool1', 'conv2', 'pool2'],
            # Create four lists to keep the features of the 4 layers separate
            ['relu2'],
            {'conv1': nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 9)),
             'selu1': nn.SELU(),
             'pool1': nn.MaxPool2d(kernel_size=(1, 2), stride=2),
             'conv2': nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 9)),
             'selu2': nn.SELU(),
             'pool2': nn.MaxPool2d(kernel_size=(1, 2), stride=2),
             'conv3': nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 9)),
             'selu3': nn.SELU()}
        )


    def forward(self, x):
        res, activation = self.features(x)
        return res, activation
    