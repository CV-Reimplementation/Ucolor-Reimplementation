# -- coding: utf-8 --
import numpy as np
import torch
import operator
import torch.nn as nn
import torch.nn.functional as F

def RGB2BGR(img):
    return torch.stack([img[:, 2, :, :], img[:, 1, :, :], img[:, 0, :, :]], dim=1)

class LAB2RGB():
    def __init__(self):
        self.M = np.array([[0.412453, 0.357580, 0.180423],
                           [0.212671, 0.715160, 0.072169],
                           [0.019334, 0.119193, 0.950227]])
        self.Mt = np.linalg.inv(self.M)

    def anti_F(self, X):
        tFX = (X - 0.137931) / 7.787
        index = X > 0.206893
        tFX[index] = torch.pow(X[index], 3)
        return tFX

    def anti_g(self, r):
        r2 = r * 12.92
        index = r > 0.0031308072830676845
        r2[index] = torch.pow(r[index], 1.0 / 2.4) * 1.055 - 0.055
        return r2

    def myPSlab2rgb(self, Lab):
        fY = (Lab[:, 0, :, :] + 16.0) / 116.0
        fX = Lab[:, 1, :, :] / 500.0 + fY
        fZ = fY - Lab[:, 2, :, :] / 200.0

        x = self.anti_F(fX)
        y = self.anti_F(fY)
        z = self.anti_F(fZ)
        x = x * 0.964221
        z = z * 0.825211

        r = 3.13405134 * x - 1.61702771 * y - 0.49065221 * z
        g = -0.97876273 * x + 1.91614223 * y + 0.03344963 * z
        b = 0.07194258 * x - 0.22897118 * y + 1.40521831 * z

        r = self.anti_g(r)
        g = self.anti_g(g)
        b = self.anti_g(b)
        return (torch.stack([r, g, b], dim=1).clamp(0.0, 1.0))
    
class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# raw pic in lab color transfer by predic matrix(sum by parm_six and parm_mt1), pixel color transfer one by one
class color_trans(object):
    def __init__(self, channel=3):
        self.split_dev = channel

    def trans(self, parm_mt2, pic_ct, mean, std):
        Lab2Rgb = LAB2RGB()
        batch_size, channel, _, _ = pic_ct.size()
        for m in range(batch_size):
            for n in range(channel):
                sd = 3 + n
                t = pic_ct[m, n, :, :]
                # color transfer: t = (t-mean[m, n, :, :])*(parm_mt2[m, sd, :, :]/std[m, n,:,:]) + parm_mt2[m, n, :, :]
                mean1 = torch.sub(t, mean[m, n, :, :])
                std_st = torch.div(parm_mt2[m, sd, :, :], std[m, n, :, :])
                out1 = torch.mul(mean1, std_st)
                out2 = torch.add(out1, parm_mt2[m, n, :, :])
                pic_ct[m, n, :, :] = out2
        pic_ct2 = Lab2Rgb.myPSlab2rgb(pic_ct)
        pic_ct3 = (RGB2BGR(pic_ct2)) / (1. / 255)

        # get pic:b*c*H*W
        return pic_ct3

    def cal_mt(self, parm_six, parm_dev_mt):
        mt1 = torch.unsqueeze(torch.unsqueeze(parm_six, -1), -1)
        parm_final_mt = mt1 + parm_dev_mt
        return parm_final_mt

    def cal_dev_mean(self, parm_six, parm_dev_mt):
        parm_six_mid = torch.unsqueeze(torch.unsqueeze(parm_six, -1), -1)
        parm_six_reshape = parm_six_mid.expand([-1, -1, 256, 256])
        mean, std = torch.split(parm_six_reshape, self.split_dev, dim=1)
        parm_final_mean_mt = mean + parm_dev_mt
        parm_final_mt = torch.cat((parm_final_mean_mt, std), 1)
        return parm_final_mt
    
class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class Basic_resnet(nn.Module):
    expansion = 1

    def __init__(self, in_channel, output_channel, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super(Basic_resnet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # 3x3 convolution with padding"
        self.conv1 = nn.Conv2d(in_channel,
                               output_channel,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               groups=groups,
                               bias=False,
                               dilation=dilation)

        self.bn1 = norm_layer(output_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(output_channel,
                               output_channel,
                               kernel_size=3,
                               stride=1,
                               padding=dilation,
                               groups=groups,
                               bias=False,
                               dilation=dilation)
        self.bn2 = norm_layer(output_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out1 = self.conv1(x)
        out2 = self.bn1(out1)
        out3 = self.relu(out2)
        out4 = self.conv2(out3)
        out = self.bn2(out4)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class Branch_L(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Branch_L, self).__init__()
        self.resnet = Basic_resnet(32, 32, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                                   norm_layer=None)
        self.Singleconv = SingleConv(n_channels, 32)
        self.Unet = Unet(32, n_classes, bilinear=bilinear)

    # input:feature map-cat of L channel and L attention map
    def forward(self, x):
        x_l = self.Singleconv(x)

        # branch to A branch and B branch,output:64 channels,
        x0 = self.resnet(x_l)

        # branch to Unet_L
        pre_mt = self.Unet(x_l)

        return x0, pre_mt


class Branch_AB(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Branch_AB, self).__init__()
        self.Doubleconv = DoubleConv(2, 32)
        self.singleconv1 = SingleConv(n_channels, 32)
        self.sigmoid = torch.nn.Sigmoid()
        self.unet = Unet(64, n_classes=n_classes, bilinear=bilinear)

    def forward(self, x, att_ab_l, L_branch):
        # make attention map(A or B) cat with L to 64 channel by conv
        map_with_L = self.Doubleconv(att_ab_l)
        map_with_L1 = self.sigmoid(map_with_L)

        # mul :(output map of A or B cat with L) to 64 * L branch resnet feature map
        cat_map = map_with_L1 * L_branch

        # x0:batch_size*64*256*256
        x0 = self.singleconv1(x)

        # map_cat:batch_size*128*256*256
        map_cat = torch.cat((cat_map, x0), 1)
        pre_mt = self.unet(map_cat)

        return pre_mt


class Regress_net(nn.Module):
    def __init__(self, num_classes=6, init_weights=False):
        super(Regress_net, self).__init__()
        self.feature = nn.Sequential(
            # conv1 image = 1/2
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # 对从上层网络Conv2d中传递下来的tensor直接进行修改
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            # conv2 image 1/4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            # conv3 image 1/8
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            # conv4 image 1/16
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            # conv5 image 1/32
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes)
        )

        if init_weights:
            self._init_weights()

    def forward(self, x):
        # x:[batch_size, 3, 256, 256]
        out1 = self.feature(x)

        out2 = torch.flatten(out1, start_dim=1)

        out3 = self.classifier(out2)

        return out3

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


class TCTLNet(nn.Module):
    def __init__(self, output_size, init_weights=False):
        super(TCTLNet, self).__init__()
        # get six parm branch
        self.regress_net = Regress_net(num_classes=6, init_weights=init_weights)

        # L branch
        self.Branch_L = Branch_L(n_channels=2, n_classes=2, bilinear=False)

        # A branch
        self.Branch_A = Branch_AB(n_channels=2, n_classes=2, bilinear=False)

        # B branch
        self.Branch_B = Branch_AB(n_channels=2, n_classes=2, bilinear=False)

        # COLOR Transfer
        self.color_trans = color_trans()
        self.output_size = output_size

    def forward(self, pic_unet, pic_vgg, raw_img, raw_grayL, raw_grayA, raw_grayB, mean, std):
        # 1.raw_img:batch_size*3*256*256, raw_greyA:batch_size*256*256, raw_greyB:batch_size*256*256, mean:batch_size*6*1*1, std:batch_size*6*1*1
        pic_L = pic_unet[:, 0, :, :].unsqueeze(1)
        pic_A = pic_unet[:, 1, :, :].unsqueeze(1)
        pic_B = pic_unet[:, 2, :, :].unsqueeze(1)
        raw_grayL = raw_grayL.unsqueeze(1)
        raw_grayA = raw_grayA.unsqueeze(1)
        raw_grayB = raw_grayB.unsqueeze(1)

        # pic_cat:[batch size, 2, 256, 256]
        pic_catL = torch.cat((pic_L, raw_grayL), 1)
        pic_catA = torch.cat((pic_A, raw_grayA), 1)
        pic_catB = torch.cat((pic_B, raw_grayB), 1)

        # cat attention map between A channel and B channel
        cat_att_AL = torch.cat((raw_grayA, raw_grayL), 1)
        cat_att_BL = torch.cat((raw_grayB, raw_grayL), 1)

        # 1.get six parm, parm_six:[batch size, 6, 1, 1]
        parm_six = self.regress_net(pic_vgg)

        # 2.raw_L to unet, parm_mtl:[batch size, 2, 256, 256]
        L_feature, parm_mtL = self.Branch_L(pic_catL)

        # 3.raw_A to unet, parm_mtA:[batch size, 2, 256, 256]
        parm_mtA = self.Branch_A(pic_catA, cat_att_AL, L_feature)

        # 4.raw_B to unet, parm_mtB:[batch size, 2, 256, 256]
        parm_mtB = self.Branch_B(pic_catB, cat_att_BL, L_feature)

        # 5.parm_mt:batch*6*256*256, [l-mean, a-mean, b-mean, l-std, a-std, b-std]
        parm_mt = torch.cat((parm_mtL[:, 0, :, :].unsqueeze(1),
                             parm_mtA[:, 0, :, :].unsqueeze(1),
                             parm_mtB[:, 0, :, :].unsqueeze(1),
                             parm_mtL[:, 1, :, :].unsqueeze(1),
                             parm_mtA[:, 1, :, :].unsqueeze(1),
                             parm_mtB[:, 1, :, :].unsqueeze(1)), 1)

        # parm_mt2:batch*6
        parm_mt_f = self.color_trans.cal_mt(parm_six, parm_mt)

        if not operator.eq(self.output_size, (256, 256)):
            parm_mt_f = get_newpic(parm_mt_f, self.output_size)
        # pic_bgr:[batch size, 3, 256, 256]
        pic_bgr = self.color_trans.trans(parm_mt_f, raw_img, mean, std)
        return parm_mt, parm_mt_f, parm_six, pic_bgr


def get_newpic(src, dst_size, align_corners=False):
    src_n, src_c, src_h, src_w = src.shape
    dst_n, dst_c, (dst_w, dst_h) = src_n, src_c, dst_size

    hd = torch.arange(0, dst_h)
    wd = torch.arange(0, dst_w)
    if align_corners:
        h = float(src_h - 1) / (dst_h - 1) * hd
        w = float(src_w - 1) / (dst_w - 1) * wd
    else:
        h = (float(src_h) / dst_h * (hd + 0.5) - 0.5).cuda()
        w = (float(src_w) / dst_w * (wd + 0.5) - 0.5).cuda()

    h = torch.clamp(h, 0, src_h - 1)
    w = torch.clamp(w, 0, src_w - 1)

    h = h.view(dst_h, 1)
    w = w.view(1, dst_w)
    h = h.repeat(1, dst_w)
    w = w.repeat(dst_h, 1)

    h0 = torch.clamp(torch.floor(h), 0, src_h - 2).cuda()
    w0 = torch.clamp(torch.floor(w), 0, src_w - 2).cuda()
    h0 = h0.long()
    w0 = w0.long()

    h1 = h0 + 1
    w1 = w0 + 1

    q00 = src[..., h0, w0]
    q01 = src[..., h0, w1]
    q10 = src[..., h1, w0]
    q11 = src[..., h1, w1]

    r0 = (w1 - w) * q00 + (w - w0) * q01
    r1 = (w1 - w) * q10 + (w - w0) * q11
    dst = (h1 - h) * r0 + (h - h0) * r1

    return dst