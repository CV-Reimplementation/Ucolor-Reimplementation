import torch
import torch.nn as nn




def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, x_HL, x_LH, x_HH

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False
    def forward(self, x):
        return dwt_init(x)


class Repoint(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Repoint, self).__init__()
        self.relu = nn.PReLU()
        self.conv = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):

        x_relu = self.relu(x)
        y = self.conv(x_relu)
        return y



def PONO(x, epsilon=1e-5):
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output, mean, std


# In the Decoder
# one can call MS(x, mean, std)
# with the mean and std are from a PONO in the encoder
def MS(x, beta, gamma):
    return x * gamma + beta


class DWT_transform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dwt = DWT()

        self.res_l = ConvINRe(in_channels)
        self.res_l2 = ConvINRe(in_channels)
        self.res_h1 = ConvBNRe(in_channels)
        self.res_h2 = ConvBNRe(in_channels)
        self.res_h3 = ConvBNRe(in_channels)
        # self.BN = nn.BatchNorm2d(3*in_channels)
        self.pa_low = PALayer_l(in_channels)
        self.ca_low = CALayer_low(in_channels)
        self.pa_high1 = PALayer_h(in_channels)
        self.ca_high1 = CALayer_high(in_channels)
        self.pa_high2 = PALayer_h(in_channels)
        self.ca_high2 = CALayer_high(in_channels)
        self.pa_high3 = PALayer_h(in_channels)
        self.ca_high3 = CALayer_high(in_channels)


        self.mean_conv1 = ConvLayer(1, 16, 1, 1)
        self.mean_conv2 = ConvLayer(16, 16, 3, 1, 2)
        self.mean_conv3 = ConvLayer(16, 1, 1, 1)

        self.std_conv1 = ConvLayer(1, 16, 1, 1)
        self.std_conv2 = ConvLayer(16, 16, 3, 1, 2)
        self.std_conv3 = ConvLayer(16, 1, 1, 1)

    def forward(self, x):
        LL, HL, LH, HH = self.dwt(x)

        LL, mean, std = PONO(LL)
        mean = self.mean_conv3(self.mean_conv2(self.mean_conv1(mean)))
        std = self.std_conv3(self.std_conv2(self.std_conv1(std)))
        LL = self.res_l(LL)
        LL = self.res_l2(LL)
        LL_ca = self.ca_low(LL)
        LL_pa = self.pa_low(LL_ca)
        LL_pa = MS(LL_pa, mean, std)
        # LL = LL_pa + LL

        # dwt_high_frequency = self.BN(dwt_high_frequency)
        HL = self.res_h1(HL)
        HL_ca = self.ca_high1(HL)
        HL_pa = self.pa_high1(HL_ca)
        # HL = HL_pa + HL

        LH = self.res_h2(LH)
        LH_ca = self.ca_high2(LH)
        LH_pa = self.pa_high2(LH_ca)
        # LH = LH_pa + LH

        HH = self.res_h3(HH)
        HH_ca = self.ca_high3(HH)
        HH_pa = self.pa_high3(HH_ca)
        # HH = HH_pa + HH

        # high_frequency = torch.cat([HL_pa, LH_pa, HH_pa], dim = 1)
        high_frequency = HL_pa + LH_pa + HH_pa

        return LL_pa, high_frequency



class PALayer_l(nn.Module):
    def __init__(self, channel):
        super(PALayer_l, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 4, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, 1, 3, padding=1, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):

        out = self.pa(x)
        y=x*out
        return y


class PALayer_h(nn.Module):
    def __init__(self, channel):
        super(PALayer_h, self).__init__()
        # self.contrast = stdv_spatials
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 4, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, 1, 3, padding=1, bias=True),
        )

    def forward(self, x):

        out = self.pa(x)
        y=x*out
        return y



class CALayer_low(nn.Module):
    def __init__(self, channel, k_size=3):
        super(CALayer_low, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class CALayer_high(nn.Module):
    def __init__(self, channel, k_size=3):
        super(CALayer_high, self).__init__()

        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

##########################################################################

def x_y(x):
    return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.ch = channels
        if self.ch == 16:
            self.relu = x_y
        else:
            self.relu = nn.PReLU()


    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) * 0.1
        out = torch.add(out, residual)
        return out

class ResidualBlock_nbn(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock_nbn, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        # self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu((self.conv1(x)))
        out = (self.conv2(out)) * 0.1
        out = torch.add(out, residual)
        return out



class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,dilation=1):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=1)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ConvBNRe(nn.Module):
    def __init__(self, channels):
        super(ConvBNRe, self).__init__()
        self.conv = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(channels)
        self.ch = channels
        if self.ch == 16:
            self.relu = x_y
        else:
            self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        y = out + residual

        return y


class ConvINRe(nn.Module):
    def __init__(self, channels):
        super(ConvINRe, self).__init__()
        self.conv = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.IN = nn.InstanceNorm2d(channels)
        self.ch = channels
        if self.ch == 16:
            self.relu = x_y
        else:
            self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.IN(out)
        out = self.relu(out)
        y = out + residual

        return y



class Waveletnet0(nn.Module):
    def __init__(self):
        super(Waveletnet0, self).__init__()
        self.num=1
        c=16
        self.pre = ConvLayer(3, c, 11, 1)

        self.conv1 = ResidualBlock(c)

        self.down1 = ConvLayer(c, 2*c, 3, 2)

        self.conv2 = ResidualBlock(2*c)

        self.down2 = ConvLayer(2*c, 4*c, 3, 2)

        self.conv3 = ResidualBlock(4 * c)
        # self.conv_bot = nn.Conv2d(8*c,8*c,3, 1, padding=1)


    def forward(self, x):

        c1 = self.pre(x)

        c1 = self.conv1(c1)

        c2_0 = self.down1(c1)

        c2 = self.conv2(c2_0)

        c3_0 = self.down2(c2)

        c3 = self.conv3(c3_0)

        return c1, c2, c3


class Waveletnet0_de(nn.Module):
    def __init__(self):
        super(Waveletnet0_de, self).__init__()
        c=16
        self.resu_bot = ResidualBlock(4 * c)
        # self.down_bot = Invertedchannel(8 * c, 4 * c, 1, 2)

        self.dwt3 = DWT_transform(4*c)
        self.up_dwt3 = nn.ConvTranspose2d(4 * c, 4 * c, kernel_size=4, stride=2, padding=1)
        self.resud3 = ResidualBlock(4 * c)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, kernel_size=4, stride=2, padding=1)

        self.rp2 = Repoint(4 * c, 2 * c)
        self.resud3_rp2 = ResidualBlock_nbn(2 * c)

        self.dwt2 = DWT_transform(2*c)
        self.up_dwt2 = nn.ConvTranspose2d(2 * c, 2 * c, kernel_size=4, stride=2, padding=1)

        self.resud2 = ResidualBlock(2 * c)
        self.up1 = nn.ConvTranspose2d(2*c, c, kernel_size=4, stride=2, padding=1)

        self.rp1 = Repoint(2 * c, c)
        self.resud2_rp1 = ResidualBlock_nbn(c)

        # self.resucat1 = ResidualBlock(2 * c)
        # self.catreduce1 = ConvLayer(2*c, c, 3, 1)

        self.dwt1 = DWT_transform(c)
        self.up_dwt1 = nn.ConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1)

        self.resud1 = ResidualBlock(c)

        self.convd0 = ConvLayer(c, 3, 3, 1)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, ori):
        c1, c2, c3 = x
        ic3 = c3

        ic3_0 = self.resu_bot(ic3)
        # ic3_0 = self.down_bot(ic3_0)

        low3, high3 = self.dwt3(ic3_0)
        ic3_0 = self.up_dwt3(low3 + high3)
        # ic3_0 = iwt(torch.cat([low3, high3], dim=1))
        ic3 = self.resud3(ic3_0)

        ic2 = self.up2(ic3)
        ic2_0 = self.rp2(torch.cat([ic2, c2],dim=1))
        # ic2_0 = self.rp2(torch.cat([ic2, c2], dim=1))

        # ic2_0 = torch.cat([ic2, c2], dim=1)
        # ic2_0 = self.resucat2(ic2_0)
        # ic2_0 = self.catreduce2(ic2_0)
        low2, high2 = self.dwt2(ic2_0)
        ic2_0 = self.up_dwt2(low2 + high2)
        # ic2_0 = iwt(torch.cat([low2, high2], dim=1))
        ic2_0 = ic2_0-ic2
        ic2 = self.resud2(ic2_0)

        ic1 = self.up1(ic2)
        ic1_0 = self.rp1(torch.cat([ic1, c1],dim=1))
        # ic1_0 = self.rp1(torch.cat([ic1, c1], dim=1))
        # ic1_0 = ic1 + c1
        # ic1_0 = torch.cat([ic1, c1], dim=1)
        # ic1_0 = self.resucat1(ic1_0)
        # ic1_0 = self.catreduce1(ic1_0)
        low1, high1 = self.dwt1(ic1_0)
        ic1_0 = self.up_dwt1(low1 + high1)
        # ic1_0 = iwt(torch.cat([low1, high1], dim=1))
        ic1_0 = ic1_0 - ic1
        ic1 = self.resud1(ic1_0)
        res = self.convd0(ic1)  #

        y_pre = res + ori

        return [y_pre, res, ic1, ic2, ic3]


class Waveletnet(nn.Module):
    def __init__(self):
        super(Waveletnet, self).__init__()
        self.num = 1
        c = 16
        self.pre = ConvLayer(3, c, 11, 1)

        self.conv1 = ResidualBlock(c)

        self.down1 = ConvLayer(c, 2*c, 3, 2)

        self.conv2 = ResidualBlock(2 * c)

        self.down2 = ConvLayer(2*c, 4*c, 3, 2)

        self.conv3 = ResidualBlock(4 * c)
        # self.conv_bot = nn.Conv2d(8 * c, 8 * c, 3, 1, padding=1)

    def forward(self, x, res):
        x = x + res

        c1 = self.pre(x)

        c1 = self.conv1(c1)

        c2_0 = self.down1(c1)

        c2 = self.conv2(c2_0)

        c3_0 = self.down2(c2)

        c3 = self.conv3(c3_0)



        return [c3, c2, c1]


class Waveletnet_de(nn.Module):
    def __init__(self):
        super(Waveletnet_de, self).__init__()
        c=16

        self.rp3 = Repoint(8 * c, 4 * c)
        self.resud4_rp3 = ResidualBlock_nbn(4 * c)
        # self.resu_bot = ResidualBlock(8 * c)
        # self.down_bot = ConvLayer(8 * c, 4 * c, 3, 1)

        self.dwt3 = DWT_transform(4 * c)
        self.up_dwt3 = nn.ConvTranspose2d(4 * c, 4 * c, kernel_size=4, stride=2, padding=1)
        self.resud3 = ResidualBlock(4 * c)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, kernel_size=4, stride=2, padding=1)

        self.rp2 = Repoint(6 * c, 2 * c)
        self.resud3_rp2 = ResidualBlock_nbn(2 * c)

        # self.resucat2 = ResidualBlock(6 * c)
        # self.catreduce2 = ConvLayer(6 * c, 2 * c, 3, 1)

        self.dwt2 = DWT_transform(2 * c)
        self.up_dwt2 = nn.ConvTranspose2d(2 * c, 2 * c, kernel_size=4, stride=2, padding=1)
        self.resud2 = ResidualBlock(2 * c)
        self.up1 = nn.ConvTranspose2d(2 * c, c, kernel_size=4, stride=2, padding=1)

        self.rp1 = Repoint(3 * c, c)
        self.resud2_rp1 = ResidualBlock_nbn(c)

        # self.resucat1 = ResidualBlock(3 * c)
        # self.catreduce1 = ConvLayer(3 * c, c, 3, 1)

        self.dwt1 = DWT_transform(c)
        self.up_dwt1 = nn.ConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1)
        self.resud1 = ResidualBlock(c)

        self.convd0 = ConvLayer(c, 3, 3, 1)


    def forward(self, x, y, ori):
        c3, c2, c1, = x
        p_ic1=y[2]
        p_ic2=y[3]
        p_ic3 = y[4]

        # ic3_0 = c3 + p_ic3
        ic3_0 = self.rp3(torch.cat([c3, p_ic3],dim=1))
        # ic3 = torch.cat([c3,p_ic3], dim=1)
        # ic3_0 = self.resu_bot(ic3)
        # ic3_0 = self.down_bot(ic3_0)
        low3, high3 = self.dwt3(ic3_0)
        ic3_0 = self.up_dwt3(low3 + high3)
        # ic3_0 = iwt(torch.cat([low3, high3], dim=1))
        ic3_0 = ic3_0 - c3
        ic3 = self.resud3(ic3_0)

        ic2 = self.up2(ic3)
        # ic2_0 = ic2 + c2 + p_ic2
        ic2_0 = self.rp2(torch.cat([ic2, c2, p_ic2], dim=1))
        # ic2_0 = torch.cat([ic2, c2, p_ic2], dim=1)
        # ic2_0 = self.resucat2(ic2_0)
        # ic2_0 = self.catreduce2(ic2_0)
        low2, high2 = self.dwt2(ic2_0)
        ic2_0 = self.up_dwt2(low2 + high2)
        # ic2_0 = iwt(torch.cat([low2, high2], dim=1))
        ic2_0 = ic2_0 - ic2
        ic2 = self.resud2(ic2_0)

        ic1 = self.up1(ic2)
        # ic1_0 = ic1 + c1 + p_ic1
        ic1_0 = self.rp1(torch.cat([ic1,  c1, p_ic1], dim=1))
        # ic1_0 = torch.cat([ic1, c1, p_ic1], dim=1)
        # ic1_0 = self.resucat1(ic1_0)
        # ic1_0 = self.catreduce1(ic1_0)
        low1, high1 = self.dwt1(ic1_0)
        ic1_0 = self.up_dwt1(low1 + high1)
        # ic1_0 = iwt(torch.cat([low1, high1], dim=1))
        ic1_0 = ic1_0 - ic1
        ic1 = self.resud1(ic1_0)
        res = self.convd0(ic1)  #

        y_pre = res + ori

        return [y_pre, res, ic1, ic2, ic3]


##########################################################################
class PRWNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=48, scale_unetfeats=24, scale_orsnetfeats=16, num_cab=4, kernel_size=3, reduction=8, bias=False):
        super(PRWNet, self).__init__()

        self.en1 = Waveletnet0()
        self.en2 = Waveletnet()
        self.en3 = Waveletnet()

        self.de1 = Waveletnet0_de()
        self.de2 = Waveletnet_de()
        self.de3 = Waveletnet_de()



    def forward(self, x3_img):
        # Original-resolution Image for Stage 3
        H = x3_img.size(2)
        W = x3_img.size(3)


        # Two Patches for Stage 2
        x2top_img  = x3_img[:,:,0:int(H/2),:]
        x2bot_img  = x3_img[:,:,int(H/2):H,:]

        # Four Patches for Stage 1
        x1ltop = x2top_img[:,:,:,0:int(W/2)]
        x1rtop = x2top_img[:,:,:,int(W/2):W]
        x1lbot = x2bot_img[:,:,:,0:int(W/2)]
        x1rbot = x2bot_img[:,:,:,int(W/2):W]

        feat1_ltop = self.en1(x1ltop)
        feat1_rtop = self.en1(x1rtop)
        feat1_lbot = self.en1(x1lbot)
        feat1_rbot = self.en1(x1rbot)

        feat1_top_cat = [torch.cat((k,v), 3) for k,v in zip(feat1_ltop,feat1_rtop)]
        feat1_bot_cat = [torch.cat((k,v), 3) for k,v in zip(feat1_lbot,feat1_rbot)]

        feat1_top = self.de1(feat1_top_cat, x2top_img)
        feat1_bot = self.de1(feat1_bot_cat, x2bot_img)
        # y1_pre = torch.cat([feat1_top[0], feat1_bot[0]], 2)

        feat2_top = self.en2(x2top_img, feat1_top[1])
        feat2_bot = self.en2(x2bot_img, feat1_bot[1])
        feat2 = [torch.cat((k, v), 2) for k, v in zip(feat2_top, feat2_bot)]
        feat1_cat = [torch.cat((k, v), 2) for k, v in zip(feat1_top, feat1_bot)]

        feat2 = self.de2(feat2, feat1_cat, x3_img)
        # y2_pre = feat2[0]

        feat3 = self.en3(x3_img, feat2[1])
        feat3 = self.de3(feat3, feat2, x3_img)
        
        y3_pre = feat3[0]


        # x2_img = F.interpolate(x3_img, scale_factor=0.5)
        # x1_img = F.interpolate(x2_img, scale_factor=0.5)
        # y1_pre, y1_res, de1_ic1, de1_ic2, de1_ic3 = self.en1(x1_img)
        #
        # y1_res = F.interpolate(y1_res, scale_factor=2)
        # y1_pre = F.interpolate(y1_pre, scale_factor=4)
        # de1_ic1 = F.interpolate(de1_ic1, scale_factor=2)
        # de1_ic2 = F.interpolate(de1_ic2, scale_factor=2)
        # de1_ic3 = F.interpolate(de1_ic3, scale_factor=2)
        #
        # y2_pre, y2_res, de2_ic1, de2_ic2, de2_ic3 = self.en2(x2_img, y1_res, de1_ic1, de1_ic2, de1_ic3)
        #
        # y2_res = F.interpolate(y2_res, scale_factor=2)
        # y2_pre = F.interpolate(y2_pre, scale_factor=2)
        # de2_ic1 = F.interpolate(de2_ic1, scale_factor=2)
        # de2_ic2 = F.interpolate(de2_ic2, scale_factor=2)
        # de2_ic3 = F.interpolate(de2_ic3, scale_factor=2)
        #
        # y3_pre, y3, de3_ic1, de3_ic2, de3_ic3 = self.en3(x3_img, y2_res, de2_ic1, de2_ic2, de2_ic3)


        return y3_pre
    
if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256).cuda()
    model = PRWNet().cuda()
    res = model(t)
    print(res.shape)