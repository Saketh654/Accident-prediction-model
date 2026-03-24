import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

class Accident3DCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d(3,32,(3,7,7),stride=(1,2,2),padding=(1,3,3))
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32,64,(3,5,5),stride=(1,2,2),padding=(1,2,2))
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64,128,(3,3,3),stride=(2,2,2),padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))

        self.fc = nn.Linear(128,1)

    def forward(self,x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool3d(x,(1,2,2))

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool3d(x,(1,2,2))

        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x)

        x = x.view(x.size(0),-1)

        x = self.fc(x)

        return x