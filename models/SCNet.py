import torch
import torch.nn as nn


def PONO(x, epsilon=1e-5):
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    out = (x - mean) / std
    return out, mean, std

def MS(x, beta, gamma):
    return x * gamma + beta


class Whiten2d(nn.Module):
    def __init__(self, num_features, t=5, eps=1e-5, affine=True):
        super(Whiten2d, self).__init__()
        self.T = t
        self.eps = eps
        self.affine = affine
        self.num_features = num_features
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):

        N, C, H, W = x.size()

        # N x C x (H x W)
        in_data = x.view(N, C, -1)

        eye = in_data.data.new().resize_(C, C)
        eye = torch.nn.init.eye_(eye).view(1, C, C).expand(N, C, C)

        # calculate other statistics
        # N x C x 1
        mean_in = in_data.mean(-1, keepdim=True)
        x_in = in_data - mean_in
        # N x C x C
        cov_in = torch.bmm(x_in, torch.transpose(x_in, 1, 2)).div(H * W)
        # N  x c x 1
        mean = mean_in
        cov = cov_in + self.eps * eye

        # perform whitening using Newton's iteration
        Ng, c, _ = cov.size()
        P = torch.eye(c).to(cov).expand(Ng, c, c)
        # reciprocal of trace of covariance
        rTr = (cov * P).sum((1, 2), keepdim=True).reciprocal_()
        cov_N = cov * rTr
        for k in range(self.T):
            P = torch.baddbmm(1.5, P, -0.5, torch.matrix_power(P, 3), cov_N)
            # P = torch.baddbmm(P, torch.matrix_power(P, 3), 1.5, -0.5, cov_N)
        # whiten matrix: the matrix inverse of covariance, i.e., cov^{-1/2}
        wm = P.mul_(rTr.sqrt())

        x_hat = torch.bmm(wm, in_data - mean)
        x_hat = x_hat.view(N, C, H, W)
        if self.affine:
            x_hat = x_hat * self.weight.view(1, self.num_features, 1, 1) + \
                self.bias.view(1, self.num_features, 1, 1)

        return x_hat
    
class SELayer(torch.nn.Module):
    def __init__(self, num_filter):
        super(SELayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv_double = torch.nn.Sequential(
            torch.nn.Conv2d(num_filter, num_filter // 16, 1, 1, 0, bias=True),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(num_filter // 16, num_filter, 1, 1, 0, bias=True),
            torch.nn.Sigmoid())

    def forward(self, x):
        mask = self.global_pool(x)
        mask = self.conv_double(mask)
        x = x * mask
        return x


class ResBlock(nn.Module):
    def __init__(self, num_filter):
        super(ResBlock, self).__init__()
        body = []
        for i in range(2):
            body.append(nn.ReflectionPad2d(1))
            body.append(nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=0))
            if i == 0:
                body.append(nn.LeakyReLU(0.2))
        body.append(SELayer(num_filter))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        x = res + x
        return x


class Up(nn.Module):
    def __init__(self):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=0),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=0),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_in = ConvBlock(ch_in=3, ch_out=64)
        self.conv1 = ConvBlock(ch_in=64, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=64)
        self.conv3 = ConvBlock(ch_in=64, ch_out=64)
        self.conv4 = ConvBlock(ch_in=64, ch_out=64)
        self.IW1 = Whiten2d(64)
        self.IW2 = Whiten2d(64)
        self.IW3 = Whiten2d(64)
        self.IW4 = Whiten2d(64)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv_in(x)

        x1, x1_mean, x1_std = PONO(x)
        x1 = self.conv1(x)
        x2 = self.pool(x1)

        x2, x2_mean, x2_std = PONO(x2)
        x2 = self.conv2(x2)
        x3 = self.pool(x2)

        x3, x3_mean, x3_std = PONO(x3)
        x3 = self.conv3(x3)
        x4 = self.pool(x3)

        x4, x4_mean, x4_std = PONO(x4)
        x4 = self.conv4(x4)

        x4_iw = self.IW4(x4)
        x3_iw = self.IW3(x3)
        x2_iw = self.IW2(x2)
        x1_iw = self.IW1(x1)

        return x1_iw, x2_iw, x3_iw, x4_iw, x1_mean, x2_mean, x3_mean, x4_mean, x1_std, x2_std, x3_std, x4_std


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.encoder = Encoder()
        self.UpConv4 = ConvBlock(ch_in=64, ch_out=64)
        self.Up3 = Up()
        self.UpConv3 = ConvBlock(ch_in=128, ch_out=64)
        self.Up2 = Up()
        self.UpConv2 = ConvBlock(ch_in=128, ch_out=64)
        self.Up1 = Up()
        self.UpConv1 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_u4 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.conv_s4 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.conv_u3 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.conv_s3 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.conv_u2 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.conv_s2 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.conv_u1 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.conv_s1 = nn.Conv2d(1, 64, kernel_size=1, padding=0)

        out_conv = []
        for i in range(1):
            out_conv.append(ResBlock(64))
        out_conv.append(nn.ReflectionPad2d(1))
        out_conv.append(nn.Conv2d(64, 3, kernel_size=3, padding=0))
        self.out_conv = nn.Sequential(*out_conv)

    def forward(self, Input):
        x1, x2, x3, x4, x1_mean, x2_mean, x3_mean, x4_mean, x1_std, x2_std, x3_std, x4_std = self.encoder(Input)

        # x4->x3
        x4_mean = self.conv_u4(x4_mean)
        x4_std = self.conv_s4(x4_std)
        x4 = MS(x4, x4_mean, x4_std)
        x4 = self.UpConv4(x4)
        d3 = self.Up3(x4)
        # x3->x2
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.UpConv3(d3)
        x3_mean = self.conv_u3(x3_mean)
        x3_std = self.conv_s3(x3_std)
        d3 = MS(d3, x3_mean, x3_std)
        d2 = self.Up2(d3)
        # x2->x1
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.UpConv2(d2)
        x2_mean = self.conv_u2(x2_mean)
        x2_std = self.conv_s2(x2_std)
        d2 = MS(d2, x2_mean, x2_std)
        d1 = self.Up1(d2)
        # x1->out
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.UpConv1(d1)
        x1_mean = self.conv_u1(x1_mean)
        x1_std = self.conv_s1(x1_std)
        d1 = MS(d1, x1_mean, x1_std)
        out = self.out_conv(d1)

        return out


class SCNet(nn.Module):
    def __init__(self):
        super(SCNet, self).__init__()
        self.decoder = Decoder()

    def forward(self, Input):
        return self.decoder(Input)


if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256).cuda()
    model = SCNet().cuda()
    res = model(t)
    print(res.shape)