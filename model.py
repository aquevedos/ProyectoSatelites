import torch.nn as nn
from torchvision import models

'''
This file defines a U-Net model for segmentation, using a pre-trained ResNet-50 as the encoder 
and a series of transpose convolutions for the decoder to generate the final segmented output.
'''

class UNet(nn.Module):
    def __init__(self, num_classes, image_size= 300):
        super(UNet, self).__init__()
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.base_layers = list(self.encoder.children())[:-2]
        self.encoder = nn.Sequential(*self.base_layers)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv_out = nn.Conv2d(32, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.conv_out(x)
        return self.upsample(x)