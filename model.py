import torch.nn as nn
from torchvision import models

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.base_layers = list(self.encoder.children())[:-2]  # Quita la capa FC y el avgpool
        self.encoder = nn.Sequential(*self.base_layers)

        # Decoder adaptado a ResNet50 (2048 canales en la última capa)
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

        # Capa de salida con las clases
        self.conv_out = nn.Conv2d(32, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.encoder(x)  # Pasamos por el encoder ResNet50
        x = self.decoder(x)  # Pasamos por el decoder
        x = self.conv_out(x)  # Mapeamos al número de clases
        return self.upsample(x)  # Ajustamos al tamaño de salida (512x512)