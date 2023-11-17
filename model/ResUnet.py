import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from torchsummary import summary
import os

# os.environ['CUDA_VISIBLE_DEVICES']='0'


def downsample():
    return nn.MaxPool3d(kernel_size=2, stride=2)
    

class Resconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resconv, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(out_channels)
   
    
    def forward(self, x):
        residual = self.bn3(self.conv1x1(x))
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = out + residual
        out = self.relu(out)
        return out
    
class ResUnet(nn.Module):
    def __init__(self, classes=2, in_channels=1, init_features=16):
        super(ResUnet,self).__init__()

        features = init_features
        self.encoder1 = Resconv(in_channels, features)
        self.pool1 = downsample()
        self.encoder2 = Resconv(features, features*2) 
        self.pool2 = downsample()
        self.encoder3 = Resconv(features*2, features*4)
        self.pool3 = downsample()
        self.encoder4 = Resconv(features*4, features*8)
        self.pool4 = downsample()
        self.bridge = Resconv(features*8, features*16)

        self.up4 =  nn.ConvTranspose3d(features*16, features*8, kernel_size=2, stride=2)
        self.decoder4 = Resconv(features*16, features*8)
        self.up3 =  nn.ConvTranspose3d(features*8, features*4, kernel_size=2, stride=2)
        self.decoder3 = Resconv(features*8, features*4)
        self.up2 =  nn.ConvTranspose3d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder2 = Resconv(features*4, features*2)
        self.up1 =  nn.ConvTranspose3d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = Resconv(features*2, features)

        self.conv = nn.Conv3d(features, out_channels=classes, kernel_size=1)
     


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bridge = self.bridge(self.pool4(enc4))

        dec4 = self.up4(bridge)
        dec4 = torch.cat((dec4, enc4),dim=1)
        dec4 =self.decoder4(dec4)
        dec3 = self.up3(dec4)
        dec3 = torch.cat((dec3, enc3),dim=1)
        dec3 =self.decoder3(dec3)
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2),dim=1)
        dec2 =self.decoder2(dec2)
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1),dim=1)
        dec1 =self.decoder1(dec1)

        outputs = self.conv(dec1)
        final = torch.sigmoid(outputs)

        return dec3, final
        
# net = ResUnet(classes=2, in_channels=1, init_features=16).cuda()
# summary(net, (1,80,128,128))




    
        




        



    


