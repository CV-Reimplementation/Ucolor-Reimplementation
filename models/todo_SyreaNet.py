import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.layers_count = 0
        for i, (in_ch, out_ch) in enumerate(zip(in_channels, out_channels)):
            setattr(self, "linear_"+str(i), nn.Linear(in_ch, out_ch))
            self.layers_count += 1

    def forward(self, x):
        for i in range(self.layers_count):
            layer = getattr(self, "linear_" + str(i))
            x = layer(x)
        return x
    
class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.BatchNorm2d(out_ch),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.BatchNorm2d(out_ch),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x
    
class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(DownConv, self).__init__()
        self.downconv = nn.Sequential(
            nn.MaxPool2d(2), # 1/2 size
            DoubleConv(in_ch, out_ch, kernel_size, stride, padding)
        )
    
    def forward(self, x):
        x = self.downconv(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, bilinear=False):
        super(UpConv, self).__init__()
        
        if bilinear:
            self.up = nn.UpSample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch1, in_ch1, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_ch1+in_ch2, out_ch)
    
    def forward(self, x1, x2):
        # note that x2's (h2, w2) is larger than x1's (h1, w1)
        x1 = self.up(x1)
        
        # input BCHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2))
        x = torch.cat([x1, x2], dim=1)  # concatenate in channel axis
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)  # TODO: kernel size 1x1 ?
    
    def forward(self, x):
        x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.layers_count = 0
        for i, (in_ch, out_ch) in enumerate(zip(in_channels, out_channels)):
            if i == 0:
                setattr(self, "encode_"+str(i), DoubleConv(in_ch, out_ch))
            else:
                setattr(self, "encode_"+str(i), DownConv(in_ch, out_ch))
            self.layers_count += 1
    
    def forward(self, x):
        outs = []
        for i in range(self.layers_count):
            layer = getattr(self, "encode_"+str(i))
            x = layer(x)
            outs.append(x)
        
        outs.reverse()
        
        return outs

class Decoder(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(Decoder, self).__init__()
        self.sigmoid = nn.Sigmoid()
        for i, (in_ch1, in_ch2, out_ch) in enumerate(zip(in_channels1, in_channels2, out_channels)):
            setattr(self, "decode_"+str(i), UpConv(in_ch1, in_ch2, out_ch))
            
    
    def forward(self, xs):
        x = self.sigmoid(xs[0])
        outs = []
        for i, xi in enumerate(xs[1:]):
            layer = getattr(self, "decode_"+str(i))
            x = layer(x, xi)
            outs.append(x)
            
        return x, outs


def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH

class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.MaxPool = nn.MaxPool2d(2)
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.MaxPool(x), self.LL(x), self.LH(x), self.HL(x), self.HH(x)

class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.UP = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)

    def forward(self, out, LH, HL, HH, LL=None, original=None):
        if LL is not None and original is not None:
            return torch.cat([self.UP(out), self.LL(LL), original], dim=1)
        else:
            return torch.cat([self.UP(out), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)

class WaveEncoder(nn.Module):
    def __init__(self, option_unpool='cat5'):
        super(WaveEncoder, self).__init__()
        self.option_unpool = option_unpool

        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1 = DoubleConv(3, 64)
        self.pool1 = WavePool(64)

        self.conv2 = DoubleConv(64*5, 128)
        self.pool2 = WavePool(128)

        self.conv3 = DoubleConv(128*5, 256)
        # self.conv3_2 = DoubleConv(256, 256)
        self.pool3 = WavePool(256)

        self.conv4 = DoubleConv(256*5, 512)

    def forward(self, x):
        skips = {}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
        return x, skips

    def encode(self, x, skips, level):
        assert level in {1, 2, 3, 4}
        if self.option_unpool == 'cat5':
            if level == 1:
                out = self.conv1(self.conv0(x))
                skips['conv1'] = out
                return out
            elif level == 2:
                Max, LL, LH, HL, HH = self.pool1(x)
                skips['pool1'] = [Max, LL, LH, HL, HH]
                out = torch.cat(skips['pool1'], dim=1)
                out = self.conv2(out)
                skips['conv2'] = out
                return out
            elif level == 3:
                Max, LL, LH, HL, HH = self.pool2(x)
                skips['pool2'] = [Max, LL, LH, HL, HH]
                out = torch.cat(skips['pool2'], dim=1)
                out = self.conv3(out)
                skips['conv3'] = out
                return out
            else:
                Max, LL, LH, HL, HH = self.pool3(x)
                skips['pool3'] = [Max, LL, LH, HL, HH]
                out = torch.cat(skips['pool3'], dim=1)
                out = self.conv4(out)
                skips['conv4'] = out
                return out
        else:
            raise NotImplementedError

class WaveDecoder(nn.Module):
    def __init__(self, option_unpool='cat5'):
        super(WaveDecoder, self).__init__()
        self.option_unpool = option_unpool
        
        self.pad = nn.ReflectionPad2d(1)
        self.conv4 = DoubleConv(512, 256)

        self.recon_block3 = WaveUnpool(256, option_unpool)
        if option_unpool == 'cat5':
            self.conv3_2 = nn.Conv2d(256*5, 256, 3, 1, 0)
        else:
            self.conv3_2 = nn.Conv2d(256*3, 256, 3, 1, 0)
        self.conv3_1 = DoubleConv(256, 128)

        self.recon_block2 = WaveUnpool(128, option_unpool)
        
        if option_unpool == 'cat5':
            self.conv2_2 = nn.Conv2d(128*5, 128, 3, 1, 0)
        else:
            self.conv2_2 = nn.Conv2d(128*3, 128, 3, 1, 0)
        self.conv2_1 = DoubleConv(128, 64)

        self.recon_block1 = WaveUnpool(64, option_unpool)
        if option_unpool == 'cat5':
            self.conv1_3 = nn.Conv2d(64*5, 64, 3, 1, 0)
        else:
            self.conv1_3 = nn.Conv2d(64*3, 64, 3, 1, 0)
        self.conv1_2 = DoubleConv(64, 32)
        self.conv1_1 = OutConv(32, 3)

    def forward(self, x, skips):
        outs = {}
        for level in [4, 3, 2, 1]:
            x, tmp = self.decode(x, skips, level)
            outs[level] = tmp
        return x, outs

    def decode(self, x, skips, level):
        assert level in {4, 3, 2, 1}
        if level == 4:
            out = self.conv4(x)
            tmp = out
            Max, LL, LH, HL, HH = skips['pool3']
            original = skips['conv3']
            if self.option_unpool == 'cat5':
                out = self.recon_block3(out, LH, HL, HH, original=original)
            else:
                out = self.recon_block3(out, LL, HL, HH, LL=LL, original=original)
            return out, tmp
        elif level == 3:
            out = self.conv3_2(self.pad(x))
            tmp = out
            out = self.conv3_1(out)
            Max, LL, LH, HL, HH = skips['pool2']
            original = skips['conv2']
            if self.option_unpool == 'cat5':
                out = self.recon_block2(out, LH, HL, HH, original=original)
            else:
                out = self.recon_block2(out, LL, HL, HH, LL=LL, original=original)
            return out, tmp
        elif level == 2:
            out = self.conv2_2(self.pad(x))
            tmp = out
            out = self.conv2_1(out)
            Max, LL, LH, HL, HH = skips['pool1']
            original = skips['conv1']
            if self.option_unpool == 'cat5':
                out = self.recon_block1(out, LH, HL, HH, original=original)
            else:
                out = self.recon_block1(out, LL, HL, HH, LL=LL, original=original)
            return out, tmp
        else:
            out = self.conv1_3(self.pad(x))
            tmp = out
            out = self.conv1_1(self.conv1_2(out))
            return out, tmp
        

class SyreaNet(nn.Module):
    def __init__(self):
        super(SyreaNet, self).__init__()
        self.wave_encoder = WaveEncoder('cat5')
        self.wave_decoder = WaveDecoder('cat5')
        self.decoder_b = WaveDecoder('cat3')
        self.decoder_t = WaveDecoder('cat3')
        
        # decoder for white point
        self.decoder_w1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_w2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_w3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.decoder_w4 = nn.Conv2d(64, 3, kernel_size=1, padding=0, bias=True)
        
    
    def forward(self, x):
        x, skips = self.wave_encoder(x)
        _, outs_b = self.decoder_b(x, skips)
        _, outs_t = self.decoder_t(x, skips)
        x_w1 = self.decoder_w1(x)
        x_w2 = self.decoder_w2(x_w1)
        x_w3 = self.decoder_w3(x_w2)
        # x_w = self.decoder_w4(self.avg_pool(x_w3))
        # x_w = self.decoder_w3(self.avg_pool(self.decoder_w2(self.decoder_w1(x))))
        skips['conv3'] = (skips['conv3'] - outs_b[3]) / (torch.maximum(outs_t[3] * self.avg_pool(x_w1),
                                                                                  torch.ones_like(outs_t[3] * 1e-3)))
        skips['conv2'] = (skips['conv2'] - outs_b[2]) / (torch.maximum(outs_t[2] * self.avg_pool(x_w2),
                                                                                  torch.ones_like(outs_t[2] * 1e-3)))
        skips['conv1'] = (skips['conv1'] - outs_b[1]) / (torch.maximum(outs_t[1] * self.avg_pool(x_w3),
                                                                       torch.ones_like(outs_t[1] * 1e-3)))
        x_cl, _ = self.wave_decoder(x, skips)
        
        return x_cl.clamp(0, 1)
    

if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256).cuda()
    model = SyreaNet().cuda()
    res = model(t)
    print(res.shape)