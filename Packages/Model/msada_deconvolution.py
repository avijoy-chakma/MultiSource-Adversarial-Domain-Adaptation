import torch.nn as nn
import torch.nn.functional as F
import torch
# Input: 64,3,1,64


class AccDeconvolution(nn.Module):
    def __init__(self):
        super(AccDeconvolution, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(1, 9))
        self.selu1 =  nn.SELU()
        self.unpool1 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(1, 9))
        self.selu2 = nn.SELU()
        self.unpool2 = nn.MaxUnpool2d(kernel_size=(1, 2), stride=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(1, 9))
        self.selu3 = nn.SELU()

    def forward(self, x, indices1, indices2, shape1, shape2):
        x = self.selu1(self.deconv1(x))
        x = self.unpool1(x, indices2, output_size = shape2)
        x = self.selu2(self.deconv2(x))
        x = self.unpool2(x, indices1, output_size = shape1)
        x = self.selu3(self.deconv3(x))
        
        return x
    
#         print("First Deconv: " + str(x.shape))
#         print("First Unpool: " + str(x.shape))
#         print("Second Deconv: " + str(x.shape))
#         print("Second Unpool: " + str(x.shape))
#         print("Third Deconv: " + str(x.shape))