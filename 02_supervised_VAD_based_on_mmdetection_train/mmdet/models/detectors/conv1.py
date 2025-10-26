import torch.nn as nn
import torch.nn.functional as F

class FR(nn.Module):
    def __init__(self):
        super(FR, self).__init__()   
        self.conv1 = nn.Conv2d(256, 64, kernel_size=(3,3), stride=1, padding=1)
        self.pl1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1)
        self.pl2 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=1, padding=1)
        self.pl3 = nn.MaxPool2d(kernel_size=(2,2))

        
    def forward(self, x):
        # Upsampling Submodule
        x1 = F.relu(self.conv1(x))
        p1 = self.pl1(x1)

        x2 = F.relu(self.conv2(p1))
        p2 = self.pl2(x2)

        x3 = F.relu(self.conv3(p2))
        p3 = self.pl3(x3)
        return p3