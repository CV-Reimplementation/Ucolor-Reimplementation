from math import log2
import torch,time
from torch import nn
from torch.nn import functional as F
import numpy as np

class SepConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, groups=in_channel, bias=bias)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        # 将maxpooling 与 global average pooling 结果拼接在一起
        return torch.cat((torch.max(x, 1) [0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = SepConv(in_channel=in_planes, out_channel=out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[-1, -2, -1], 
                    [0, 0, 0], 
                    [1, 2, 1]]
        kernel_h = [[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        #x = np.clip(x, 0, 1)
        return x

class SFGNet(nn.Module):
    def __init__(self, H=0, W=0, batch_size=0):
        super(SFGNet, self).__init__()
        in_channel=3
        out_channel=3
        num_dab=1
        num_rrg=4
        n_feats=64
        reduction=16
        kernel_size = 3

        self.conv1 = nn.Sequential(
            SepConv(in_channel=in_channel, out_channel=n_feats, kernel_size=kernel_size),nn.BatchNorm2d(n_feats),nn.GELU()
        )
        self.conv2 = nn.Sequential(
            SepConv(in_channel=n_feats, out_channel=n_feats, kernel_size=kernel_size),nn.BatchNorm2d(n_feats),nn.GELU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=1, padding=0,
                      stride=1),
            SepConv(in_channel=n_feats, out_channel=n_feats, kernel_size=kernel_size),nn.BatchNorm2d(n_feats),nn.GELU()
        )
        self.conv4 = nn.Sequential(
            SepConv(in_channel=n_feats, out_channel=out_channel, kernel_size=kernel_size)
        )
        self.FRDB1=FRDB(nChannels=n_feats)
        self.FRDB2=FRDB(nChannels=n_feats)
        self.FRDB3=FRDB(nChannels=n_feats)
        self.FRDB4=FRDB(nChannels=n_feats)

        self.SRDB1=SRDB(nChannels=n_feats)
        self.SRDB2=SRDB(nChannels=n_feats)
        self.SRDB3=SRDB(nChannels=n_feats)
        self.SRDB4=SRDB(nChannels=n_feats)
        self.conv5 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, padding=(1 - 1) // 2,
                              bias=False),#nn.BatchNorm2d(3),
                              nn.GELU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, padding=(1 - 1) // 2,
                              bias=False),nn.Sigmoid()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=(3 - 1) // 2,
                              bias=False), nn.Conv2d(3, 3, kernel_size=3, padding=(3 - 1) // 2,
                              bias=False), nn.Conv2d(3, 3, kernel_size=3, padding=(3 - 1) // 2,
                              bias=False), nn.Tanh()
        )
        self.conv8 = nn.Sequential(
            SepConv(in_channel=in_channel, out_channel=n_feats, kernel_size=kernel_size),nn.BatchNorm2d(n_feats),nn.GELU()
        )
        self.g=Get_gradient()
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.25)
        self.conv11 = nn.Sequential(
            SepConv(in_channel=in_channel, out_channel=n_feats, kernel_size=kernel_size),nn.BatchNorm2d(n_feats),nn.GELU()
        )
        self.conv22 = nn.Sequential(
            SepConv(in_channel=n_feats, out_channel=n_feats, kernel_size=kernel_size)
        )
        self.conv33 = nn.Sequential(
            SepConv(in_channel=n_feats, out_channel=n_feats, kernel_size=kernel_size),nn.BatchNorm2d(n_feats),nn.GELU()
        )
        self.conv44 = nn.Sequential(
            SepConv(in_channel=n_feats, out_channel=in_channel, kernel_size=kernel_size)
        )

        
        self.gamma_max = 1
        self.gamma_min = 0.1

    def forward(self, x,grad_gt=None,l_mask=0.00001):

        
        beta = 1.0
        if l_mask >= self.gamma_max:
            beta = 1.0
        elif l_mask >= self.gamma_min:
            beta = (l_mask - self.gamma_min)/(self.gamma_max - self.gamma_min)
        else:
            beta = 0.0
        
        s1 = time.time()
        lo0=x
        
        x0 = self.conv1(x)
        x0_ = self.conv2(x0)
        x00 = self.FRDB1(x0_)
        x01 = self.SRDB1(x0_)
        x1=x00+x01
        x10 = self.FRDB2(x1)
        x11 = self.SRDB2(x1)
        x2=x10+x11
        x20 = self.FRDB3(x2)
        x21 = self.SRDB3(x2)
        x3=x20+x21
        x30 = self.FRDB4(x3)
        x31 = self.SRDB4(x3)
        x4=x30+x31
        xo1 = self.conv3(x4+x0_)
        xo = self.conv4(xo1+x0)

        out1=xo+x
        xo=out1
        #out1=xo
        grad=self.g(xo).cuda()
        grad=self.conv5(grad)
        grad =self.conv6(grad)
        grad =self.conv7(grad)
        if grad_gt is None:
            mgrad = grad
        else:
            mgrad = beta*grad_gt + (1-beta)*grad
        ggrad =self.conv8(mgrad)
        xou1 = self.conv11(xo)
        xou2 = self.conv22(xou1)
        xout = self.alpha * ggrad * xou2 + (1-self.alpha) * xou2
        xou3 = self.conv33(xout+xou2)
        xout = self.conv44(xou3+xou1)
        xout = xout + xo 

        return out1

class make_fdense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=1):
        super(make_fdense, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False),nn.BatchNorm2d(growthRate)
        )
        self.bat = nn.BatchNorm2d(growthRate),
        self.leaky=nn.LeakyReLU(0.1,inplace=True)

    def forward(self, x):
        out = self.leaky(self.conv(x))
        out = torch.cat((x, out), 1)
        return out
    
class FRDB(nn.Module):
    def __init__(self, nChannels, nDenselayer=2, growthRate=32):
        super(FRDB, self).__init__()
        nChannels_1 = nChannels
        nChannels_2 = nChannels
        modules1 = []
        for i in range(nDenselayer):
            modules1.append(make_fdense(nChannels_1, growthRate))
            nChannels_1 += growthRate
        self.dense_layers1 = nn.Sequential(*modules1)
        modules2 = []
        for i in range(nDenselayer):
            modules2.append(make_fdense(nChannels_2, growthRate))
            nChannels_2 += growthRate
        self.dense_layers2 = nn.Sequential(*modules2)
        self.conv_1 = nn.Conv2d(nChannels_1, nChannels, kernel_size=1, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(nChannels_2, nChannels, kernel_size=1, padding=0, bias=False)
        


    def forward(self, x):
        _, _, H, W = x.shape
        #print(x.shape)
        x_freq = torch.fft.rfft2(x, norm='backward')
        #print(x_freq.shape)
        mag = torch.abs(x_freq)
        #print(mag.shape)
        pha = torch.angle(x_freq)
        mag = self.dense_layers1(mag)
        #print(mag.shape)
        mag = self.conv_1(mag)
        #print(mag.shape)
        pha = self.dense_layers2(pha)
        pha = self.conv_2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        out = out + x
        return out


class SRDB(nn.Module):
    def __init__(self, nChannels, growthRate=64):
        super(SRDB, self).__init__()
        nChannels_ = nChannels
        modules1 = []
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=1, padding=(1 - 1) // 2,
                              bias=False)
        self.conv2 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=(3 - 1) // 2,
                              bias=False)
        self.conv3 = nn.Conv2d(nChannels, growthRate, kernel_size=5, padding=(5 - 1) // 2,
                              bias=False)
        
        self.conv11 = nn.Conv2d(nChannels, growthRate, kernel_size=1, padding=(1 - 1) // 2,
                              bias=False)
        self.conv22 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=(3 - 1) // 2,
                              bias=False)
        self.conv33 = nn.Conv2d(nChannels, growthRate, kernel_size=5, padding=(5 - 1) // 2,
                              bias=False)
        
        self.conv4 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=(3 - 1) // 2,
                              bias=False)
        self.conv5 = nn.Conv2d(nChannels, growthRate, kernel_size=5, padding=(5 - 1) // 2,
                              bias=False)
        self.conv6 = nn.Conv2d(nChannels, growthRate, kernel_size=1, padding=(1 - 1) // 2,
                              bias=False)
        self.leaky1=nn.LeakyReLU(0.1,inplace=True)
        self.leaky2=nn.LeakyReLU(0.1,inplace=True)
        self.leaky3=nn.LeakyReLU(0.1,inplace=True)
        self.bat1 = nn.BatchNorm2d(nChannels),
        self.bat2 = nn.BatchNorm2d(nChannels),
        self.bat3 = nn.BatchNorm2d(nChannels),
        self.bat4 = nn.BatchNorm2d(nChannels),
        self.bat5 = nn.BatchNorm2d(nChannels),
        


    def forward(self, x):
       
        x_1= self.leaky1(self.conv1(x))
        x_2= self.leaky2(self.conv2(x))
        x_3= self.leaky3(self.conv3(x))

        x_11=x_1+x_3
        x_22=x_1+x_3+x_2
        x_33=x_2+x_3

        x_111= self.conv11(x_11)
        x_222= self.conv22(x_22)
        x_333= self.conv33(x_33)

        x_0=x_111+x+x_222+x_333
        x_0=self.conv6(x_0)

        out = x_0 + x
        return out

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)







def gen_att(haze,clear):
    r = haze[:, 0, :, :].unsqueeze(1)
    g = haze[:, 1, :, :].unsqueeze(1)
    b = haze[:, 2, :, :].unsqueeze(1)
    Y = 0.299 * r + 0.587 * g + 0.144 * b
    r_clear = clear[:, 0, :, :].unsqueeze(1)
    g_clear = clear[:, 1, :, :].unsqueeze(1)
    b_clear = clear[:, 2, :, :].unsqueeze(1)
    Y_clear = 0.299 * r_clear + 0.587 * g_clear + 0.144 * b_clear
    m_g = Y - Y_clear
    m_g_max = torch.max(torch.max(m_g,2).values,2).values.unsqueeze(-1).unsqueeze(-1)+1e-6
    m_g_min = torch.min(torch.min(m_g,2).values,2).values.unsqueeze(-1).unsqueeze(-1)
    m_g_l = (m_g- m_g_min)/(m_g_max-m_g_min)
    # s = haze - clear
    return m_g_l



from torchvision.models.vgg import vgg16
class Vgg(nn.Module):
      def __init__(self):
        super(Vgg, self).__init__()
        features = vgg16(torch.load("vgg16-397923af.pth")).features.cuda()
        #self.aeatures = vgg16(torch.load("vgg16-397923af.pth")).features.cuda()
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])

        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

      def forward(self, x):
        #a=self.aeatures(x)
        #print(a.shape)
        h = self.to_relu_1_2(x)
        #print(h.shape)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h

        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3)
        return out

      def generate_visuals_for_evaluation(self, data, mode):
        with torch.no_grad():
            visuals = {}
            AtoB = self.opt.direction == "AtoB"
            G = self.netG_A
            source = data["A" if AtoB else "B"].to(self.device)
            if mode == "forward":
                visuals["fake_B"] = G(source)
            else:
                raise ValueError("mode %s is not recognized" % mode)
            return visuals





def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output

from math import exp
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)


class AFFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(AFFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        
        #print(x.shape)
        #x_freq = torch.fft.rfft2(x, norm='backward')
        #print(x_freq.shape)
        pred_fft = torch.abs(pred_fft)
        target_fft = torch.abs(target_fft)
        #print(mag.shape)
        #pha = torch.angle(x_freq)
        #mag = self.dense_layers1(mag)
        #print(mag.shape)
        #mag = self.conv_1(mag)
        #print(mag.shape)
        #pha = self.dense_layers2(pha)
        #pha = self.conv_2(pha)
        #real = mag * torch.cos(pha)
        #imag = mag * torch.sin(pha)
        #x_out = torch.complex(real, imag)
        #out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        #pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        #target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)
    

if __name__ == '__main__':
    inp = torch.randn(1, 3, 256, 256).cuda()
    model = SFGNet().cuda()
    res = model(inp)
    print(res.shape)