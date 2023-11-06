import torch
import torch.nn as nn



#################################################################################################################
# Base UNet convolutional encoder block
#################################################################################################################
class UNetEncoderConvBlock (nn.Module):
    def __init__(self, in_c, out_c):
        super(UNetEncoderConvBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=out_c),
            nn.Conv2d(out_c, out_c, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.max_pooling_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        return


    def forward(self, img):
        res = self.conv_block(img)
        out = self.max_pooling_2x2(res)

        return out, res
#################################################################################################################



#################################################################################################################
# Base UNet bottleneck block
#################################################################################################################
class UNetBottleneckBlock (nn.Module):
    def __init__(self, in_c, out_c):
        super(UNetBottleneckBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=out_c),
            nn.Conv2d(out_c, out_c, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        return

    
    def forward(self, img):
        return self.conv_block(img)    
#################################################################################################################



#################################################################################################################
# Base UNet convolutional decoder block
#################################################################################################################
class UNetDecoderConvBlock (nn.Module):
    def __init__(self, in_c, out_c):
        super(UNetDecoderConvBlock, self).__init__()

        self.up_conv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=out_c),
            nn.Conv2d(out_c, out_c, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        return


    def forward(self, img, residual):
        img = self.up_conv(img)
        img = self.conv_block(torch.cat([self._center_crop_(residual, img), img], dim=1))

        return img
    

    def _center_crop_ (self, src, trg):
        indent = (src.size()[2] - trg.size()[2]) // 2

        if src.size()[2] % 2 == 0:
            return src[:, :, indent:-indent, indent:-indent]
        else:
            return src[:, :, indent:-indent-1, indent:-indent-1]
#################################################################################################################



                                    ######################################
                                    #     _      ____      _      ____   #
                                    #    / \    / ___|    / \    | __ )  #
                                    #   / _ \  | |       / _ \   |  _ \  #
                                    #  / ___ \ | |___ _ / ___ \ _| |_) | #
                                    # /_/   \_(_)____(_)_/   \_(_)____/  #
                                    ######################################



#################################################################################################################
# Vanilla UNet architechture implementation
# This version smaller than original one, which has 4 encoder and decoder blocks
#################################################################################################################
class UNet (nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.encoder = nn.ModuleList([
            UNetEncoderConvBlock(3, 64),
            UNetEncoderConvBlock(64, 128),
            UNetEncoderConvBlock(128, 256),
        ])

        self.bottleneck = UNetBottleneckBlock(256, 512)

        self.decoder = nn.ModuleList([
            UNetDecoderConvBlock(512, 256),
            UNetDecoderConvBlock(256, 128),
            UNetDecoderConvBlock(128, 64)
        ])

        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

        return
    

    def forward (self, img):
        residuals = []

        for e_block in self.encoder:
            img, res = e_block(img)
            residuals.append(res)

        img = self.bottleneck(img)

        for i, d_block in enumerate(self.decoder):
            img = d_block(img, residuals[-i-1])

        img = self.final_layer(img)

        return img
#################################################################################################################