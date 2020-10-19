import torch
import torch.nn as nn


class UNet(nn.Module):
    
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU()
        )
        
        self.drop = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(),
        )
        
        self.up6 = nn.Sequential(
            nn.Conv2d(1024, 512, 2),
            nn.ReLU(),
            nn.Upsample(size=(2, 2))
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU()
        )
        
        self.up7 = nn.Sequential(
            nn.Conv2d(512, 256, 2),
            nn.ReLU(),
            nn.Upsample(size=(2, 2))
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )
        
        self.up8 = nn.Sequential(
            nn.Conv2d(256, 128, 2),
            nn.ReLU(),
            nn.Upsample(size=(2, 2))
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        
        self.up9 = nn.Sequential(
            nn.Conv2d(128, 64, 2),
            nn.ReLU(),
            nn.Upsample(size=(2, 2))
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding=1),
            nn.ReLU()
        )
        
        self.conv10 = nn.Conv2d(2, 1, 1)
    
    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool(conv3)
        
        conv4 = self.conv4(pool3)
        drop4 = self.drop(conv4)
        pool4 = self.pool(drop4)
        
        conv5 = self.conv5(pool4)
        drop5 = self.drop(conv5)
        
        up6 = self.up6(drop5)
        merge6 = torch.cat((drop4, up6), 1)
        conv6 = self.conv6(merge6)
        
        up7 = self.up7(conv6)
        merge7 = torch.cat((conv3, up7), 1)
        conv7 = self.conv7(merge7)
        
        up8 = self.up8(conv7)
        merge8 = torch.cat((conv2, up8), 1)
        conv8 = self.conv8(merge8)
        
        up9 = self.up9(conv8)
        merge9 = torch.cat((conv1, up9), 1)
        conv9 = self.conv9(merge9)
        
        conv10 = self.conv10(conv9)
        
        out = torch.sigmoid(conv10)
        return out
        
        
        