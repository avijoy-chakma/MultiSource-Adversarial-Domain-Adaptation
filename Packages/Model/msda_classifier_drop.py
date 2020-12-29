# encoding=utf-8
"""
    Created on 10:41 2018/11/10 
    @author: Avijoy Chakma
"""
import torch.nn as nn
import torch.nn.functional as F


class AccClassifier(nn.Module):
    def __init__(self, gt_size):
        super(AccClassifier, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=256*18, out_features=128),
            nn.SELU(),
            nn.AlphaDropout(0.5)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=gt_size)
        )

    def forward(self, x):
        #11 9 7 = 13
        # 9 9 9 = 12
        out = x.reshape(-1, 256*18)
#         out = x.reshape(-1, 512*10)
        out = self.fc1(out)
        out = self.fc2(out)
        return out