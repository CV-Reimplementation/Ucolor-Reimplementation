import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class S_D(nn.Module):
    def __init__(self):
        super(S_D, self).__init__()
        self.A = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.B = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=5, dilation=5)
        self.LReLU = nn.LeakyReLU(0.2, inplace=True)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        A1 = self.LReLU(self.A(x))
        A2 = self.LReLU(self.B(x))
        A3 = A1 - A2
        Guided_map = self.Sigmoid(A3)
        Detail_map = x * Guided_map
        Structure_map = x - Detail_map

        return Structure_map, Detail_map


class simam_module(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)

        return s

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return self.activaton(y)


class SD_3D(nn.Module):
    def __init__(self):
        super(SD_3D, self).__init__()
        self.SD = S_D()
        self.sim = simam_module()

    def forward(self, x):
        Structure_map, Detail_map = self.SD(x)
        M1_0 = self.sim(Structure_map)
        M2_0 = M1_0 * Detail_map
        M1_1 = self.sim(M2_0)
        M2_1 = M1_1 * Detail_map
        M1_2 = self.sim(M2_1)
        M2_2 = M1_2 * Detail_map

        return M2_2, M2_1, M2_0


class BOths(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(BOths, self).__init__()
        # Conv
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=True)
        # Function
        self.LReLU = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.In1 = nn.InstanceNorm2d(16)
        # SD_3D
        self.SD3D = SD_3D()
        # Gate
        self.gate = nn.Conv2d(16 * 3, 3, 3, 1, 1, bias=True)
        # Final
        self.Final = nn.Conv2d(19, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        en = self.conv1(x)  # 16, 256, 256
        en = self.LReLU(en)
        en = self.In1(en)
        # SD3D
        sd2, sd1, sd0 = self.SD3D(en)
        # Gate
        gates = self.gate(torch.cat((sd0, sd1, sd2), dim=1))
        gated_X = sd0 * gates[:, [0], :, :] + sd1 * gates[:, [1], :, :] + sd2 * gates[:, [2], :, :]  # 16, 256, 256
        gated_X = torch.cat((gated_X, x), dim=1)  # 19, 256, 256
        # Final
        result = self.Final(gated_X)

        return torch.tanh(result)
        # 256*256*3

if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256).cuda()
    model = BOths().cuda()
    res = model(t)
    print(res.shape)