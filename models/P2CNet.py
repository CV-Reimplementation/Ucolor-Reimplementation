import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from timm.models.layers import trunc_normal_


class Volume_2D(nn.Module):
    def __init__(self, indim=256):
        super(Volume_2D, self).__init__()
        self.c2f_dim = 33
        self.conv_dim = indim
        coarse_range = torch.arange(-1., 1. + 0.01, step=0.0625).reshape(1, -1)  # 1 k
        self.color_range = nn.Parameter(coarse_range[None, :, :, None, None], requires_grad=False)  # 1 1 k 1 1
        self.coarse_conv = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, kernel_size=1, stride=1, padding=0),
            nn.Hardswish(inplace=True),
            nn.Conv2d(self.conv_dim, 2 * self.c2f_dim, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        cost_feature = self.coarse_conv(x).view(b, 2, self.c2f_dim, h, w)
        prob = F.softmax(cost_feature, dim=2)  # b 2 k h w
        exp = torch.sum(prob * self.color_range, dim=2)  # b 2 h w
        return exp


class ColorCompenateNet(nn.Module):
    def __init__(self, cont_dim=64, color_dim=64):
        super(ColorCompenateNet, self).__init__()
        self.color_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1)
        )
        self.context_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1)
        )
        self.encoder_0 = nn.Sequential(
            nn.InstanceNorm2d(128, affine=True),
            nn.Hardswish(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True)  # 1/2
        )
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.Hardswish(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True)  # 1/4
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, groups=2),
            nn.InstanceNorm2d(256, affine=True),
            nn.Hardswish(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=2),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True)  # 1/8
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.Hardswish(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True)  # 1/16
        )
        self.color_decoder3 = Volume_2D(indim=256)
        self.color_decoder2 = Volume_2D(indim=258)
        self.color_decoder1 = Volume_2D(indim=130)
        self.color_decoder0 = Volume_2D(indim=130)
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # step1: feature extraction
        h, w = x.shape[2:]
        l = x[:, :1, :, :]
        ab = x[:, 1:, :, :]
        feat_color = self.color_encoder(ab)
        fest_cont = self.context_encoder(l)
        feat = torch.cat([feat_color, fest_cont], dim=1)
        feat0 = self.encoder_0(feat)  # 1/2 128
        h0, w0 = feat0.shape[2:]
        feat1 = self.encoder_1(feat0)  # 1/4 128
        h1, w1 = feat1.shape[2:]
        feat2 = self.encoder_2(feat1)  # 1/8 256
        h2, w2 = feat2.shape[2:]
        feat3 = self.encoder_3(feat2)  # 1/16 256

        # step2: multi-scale probabilistic volumetric fusion
        pre_ab3 = self.color_decoder3(feat3)
        pre_ab3 = F.interpolate(pre_ab3, size=(h2, w2), mode='bilinear', align_corners=True)  # 1/8 2
        feat2 = torch.cat([feat2, pre_ab3], dim=1)  # 1/8 258

        pre_ab2 = self.color_decoder2(feat2)
        pre_ab2 = F.interpolate(pre_ab2, size=(h1, w1), mode='bilinear', align_corners=True)  # 1/4 2
        feat1 = torch.cat([feat1, pre_ab2], dim=1)  # 1/4 130

        pre_ab1 = self.color_decoder1(feat1)
        pre_ab1 = F.interpolate(pre_ab1, size=(h0, w0), mode='bilinear', align_corners=True)  # 1/2 2
        feat0 = torch.cat([feat0, pre_ab1], dim=1)  # 1/2 130

        pre_ab0 = self.color_decoder0(feat0)
        pre_ab0 = F.interpolate(pre_ab0, size=(h, w), mode='bilinear', align_corners=True)  # 1 2

        return l, pre_ab0, pre_ab1, pre_ab2, pre_ab3


class P2CNet(nn.Module):
    def __init__(self, dim=64):
        super(P2CNet, self).__init__()
        self.color = ColorCompenateNet(cont_dim=dim, color_dim=dim)

    def forward(self, x):
        l, ab0, ab1, ab2, ab3 = self.color(x)
        lab = torch.cat([l * 100., ab0 * 127.], dim=1)
        rgb = kornia.color.lab_to_rgb(lab)  # 0~1
        return {'ab_pred0': ab0, 'ab_pred1': ab1, 'ab_pred2': ab2, 'ab_pred3': ab3, 'lab_rgb': rgb}


if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256).cuda()
    model = P2CNet().cuda()
    res = model(t)['lab_rgb']
    print(res.shape)