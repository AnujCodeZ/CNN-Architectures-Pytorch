import torch
import torch.nn as nn


class FCN(nn.Module):
    
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.fcn1 = nn.Sequential(
            nn.Conv2d(512, 64, 1, 1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.fcn2 = nn.Sequential(
            nn.Conv2d(64, num_classes, 1, 1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            nn.AdaptiveMaxPool2d(num_classes),
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = self.fcn1(x)
        x = self.fcn2(x)
        
        out = torch.softmax(dim=1)
        return out
        