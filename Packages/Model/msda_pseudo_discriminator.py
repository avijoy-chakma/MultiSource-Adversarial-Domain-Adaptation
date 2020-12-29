import math
import torch.nn as nn
import torch.nn.functional as F

class AccPseudoDiscriminator(nn.Module):
    def __init__(self, gt_size):
        super(AccPseudoDiscriminator, self).__init__()
        
        self.Dfc1 = nn.Sequential(
            nn.Linear(in_features=256*18, out_features=128),
            nn.SELU()
        )
        
        self.Dfl1 = nn.Sequential(
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )
        
        self.Dfl2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=gt_size)
        )
       
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, input):
        #11 9 7 = 13
        # 9 9 9 = 12
        out = input.reshape(-1, 256*18)
        out = self.Dfc1(out)
        disc_out = self.Dfl1(out)
        class_out = self.Dfl2(out)
        return disc_out, class_out