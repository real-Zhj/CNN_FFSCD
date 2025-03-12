import torch
import torch.nn as nn
import RepVGG
from torch.utils.tensorboard import SummaryWriter

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.repvgg = RepVGG.create_RepVGG_A0(deploy=False)

    def forward(self, x):
        x = self.repvgg(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # stage 1: blocks*1
        self.trans_conv1 = nn.ConvTranspose2d(in_channels=1280, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.BN1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()
        # stage 2: blocks*6
        self.trans_conv21 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.BN21 = nn.BatchNorm2d(256)
        self.relu21 = nn.ReLU()

        self.trans_conv31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.BN31 = nn.BatchNorm2d(128)
        self.relu31 = nn.ReLU()

        self.trans_conv41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.BN41 = nn.BatchNorm2d(64)
        self.relu41 = nn.ReLU()

        self.trans_conv5 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.relu1(self.BN1(self.trans_conv1(x)))
        x = self.relu21(self.BN21(self.trans_conv21(x)))
        x = self.relu31(self.BN31(self.trans_conv31(x)))
        x = self.relu41(self.BN41(self.trans_conv41(x)))
        x = self.trans_conv5(x)
        return x

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
