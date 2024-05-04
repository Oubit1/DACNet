# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super(UpBlock, self).__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling==0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling==1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode_upsampling==2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode_upsampling==3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
class UpBlock_f1(nn.Module):
    """Upssampling followed by ConvBlock"""
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super(UpBlock_f1, self).__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling==0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling==1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode_upsampling==2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode_upsampling==3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        return x1

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
 
        self.fc1_1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc1_2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.fc2_1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2_2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))

        
        x1 = self.fc1_1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc1_2(x1)
        x1 = torch.sigmoid(x1)

        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
 
        x2 = self.fc2_1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2_2(x2)
        x2 = torch.sigmoid(x2)

        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels, channelAttention_reduce=4):
        super(SpatialAttention, self).__init__()
        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)

        # 修改了Conv2d的参数，确保输入和输出通道数一致
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        # 全局感知器
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs

        # 修改Conv2d的参数，确保输入和输出通道数一致
        spatial_att = self.conv(inputs)
        out = spatial_att * inputs
        out = self.conv(out)
        return out

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class Decoder_A1(nn.Module):
    def __init__(self, params):
        super(Decoder_A1, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock_f1(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock_f1(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock_f1(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.ca4 = ChannelAttention(16,32)
        self.sa4 = SpatialAttention(16,16)
        self.up4 = UpBlock_f1(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x5 = self.up1(x4)
        x5 += x3
        x6 = self.up2(x5)
        x6 += x2
        x7 = self.up3(x6)
        x7 += x1
        x8 = self.up4(x7)
        x8 = self.ca4(x8) * x8  # 通道注意力模块
        x8 = self.sa4(x8) * x8  # 空间注意力模块
        # x8 += x0
        output = self.out_conv(x8)
        return output
class Decoder_A2(nn.Module):
    def __init__(self, params):
        super(Decoder_A2, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock_f1(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock_f1(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.ca3 = ChannelAttention(32,64)
        self.sa3 = SpatialAttention(32,32)
        self.up3 = UpBlock_f1(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock_f1(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x5 = self.up1(x4)
        x5 += x3
        x6 = self.up2(x5)
        x6 += x2
        x7 = self.up3(x6)
        x7 = self.ca3(x7) * x7  # 通道注意力模块
        x7 = self.sa3(x7) * x7  # 空间注意力模块
        # X7 += x1
        x8 = self.up4(x7)
        x8 += x0
        output = self.out_conv(x8)
        return output
class Decoder_A3(nn.Module):
    def __init__(self, params):
        super(Decoder_A3, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock_f1(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.ca2 = ChannelAttention(64,128)
        self.sa2 = SpatialAttention(64,64)
        self.up2 = UpBlock_f1(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock_f1(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock_f1(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x5 = self.up1(x4)
        x5 += x3
        x6 = self.up2(x5)
        x6 = self.ca2(x6) * x6  # 通道注意力模块
        x6 = self.sa2(x6) * x6  # 空间注意力模块
        # x6 += x2
        x7 = self.up3(x6)
        x7 += x1
        x8 = self.up4(x7)
        x8 += x0
        output = self.out_conv(x8)
        return output
class Decoder_A4(nn.Module):
    def __init__(self, params):
        super(Decoder_A4, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)
        self.ca1 = ChannelAttention(128,256)
        self.sa1 = SpatialAttention(128,128)
        self.up1 = UpBlock_f1(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up2 = UpBlock_f1(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up3 = UpBlock_f1(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, mode_upsampling=self.up_type)
        self.up4 = UpBlock_f1(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x5 = self.up1(x4)
        x5 = self.ca1(x5) * x5  # 通道注意力模块
        x5 = self.sa1(x5) * x5  # 空间注意力模块
        # X5 += x3
        x6 = self.up2(x5)
        x6 += x2
        x7 = self.up3(x6)
        x7 += x1
        x8 = self.up4(x7)
        x8 += x0
        output = self.out_conv(x8)
        return output
class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        return output1
 
class DACNet2d_v2(nn.Module):
    def __init__(self, in_chns, class_num):
        super(ACNet2d_v2, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}
        params2 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}
        params3 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}
        self.encoder1 = Encoder(params1)
        self.decoder1 = Decoder(params1)
        self.decoder2 = Decoder_f1(params2)
        self.decoder3 = Decoder_f2(params3)
        
    def forward(self, x):
        feature = self.encoder1(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        output3 = self.decoder3(feature)
        return output1, output2, output3
        
class DACNet2d_v3(nn.Module):
    def __init__(self, in_chns, class_num):
        super(ACNet2d_v3, self).__init__()

        params1 = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 0,
                  'acti_func': 'relu'}
        self.encoder1 = Encoder(params1)
        self.decoder1 = Decoder_A1(params1)
        self.decoder2 = Decoder_A2(params1)
        self.decoder3 = Decoder_A3(params1)
        self.decoder4 = Decoder_A4(params1)
        
    def forward(self, x):
        feature = self.encoder1(x)
        output1 = self.decoder1(feature)
        output2 = self.decoder2(feature)
        output3 = self.decoder3(feature)
        output4 = self.decoder4(feature)
        return output1, output2, output3, output4

if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from ptflops import get_model_complexity_info
    model = UNet(in_chns=1, class_num=4).cuda()
    with torch.cuda.device(0):
      macs, params = get_model_complexity_info(model, (1, 256, 256), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    import ipdb; ipdb.set_trace()
