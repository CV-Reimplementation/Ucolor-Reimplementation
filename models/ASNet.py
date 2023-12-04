import torch
import torch.nn as nn

class MASEblock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveMaxPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)

        return x
class MISEblock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveMaxPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = -self.squeeze(-x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)

        return x
class ANB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.maseblock = MASEblock(in_channels)
        self.miseblock = MISEblock(in_channels)

    def forward(self, x):

        im_h = self.maseblock(x)
        im_l = self.miseblock(x)

        me = torch.tensor(0.00001, dtype=torch.float32).cuda()

        x = (x - im_l) / torch.maximum(im_h - im_l, me)
        x = torch.clip(x, 0.0, 1.0)

        return x
class ASB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):

        x = self.conv(x)

        return x
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class CorlorCorrection(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            ASB(in_channels, out_channels, kernel_size=kernel_size, **kwargs),
            ANB(out_channels),
            ASB(out_channels, out_channels, kernel_size=kernel_size, **kwargs),
            ASB(out_channels, out_channels, kernel_size=kernel_size, **kwargs),
            ASB(out_channels, out_channels, kernel_size=kernel_size, **kwargs),
            ASB(out_channels, out_channels, kernel_size=kernel_size, **kwargs),
            ASB(out_channels, out_channels, kernel_size=kernel_size, **kwargs),
            ANB(out_channels),
            BasicConv2d(out_channels, in_channels, kernel_size=kernel_size, **kwargs),
           )

    def forward(self, x):

        x = self.conv(x)

        return x


class ASNet(nn.Module):

    def __init__(self):
        super(ASNet, self).__init__()

        self.conv2wb_1 = CorlorCorrection(3, 128, 3, stride=1, padding=1)

    def forward(self, img_haze):

        conv_wb1 = self.conv2wb_1(img_haze)

        return conv_wb1
    

if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256).cuda()
    model = ASNet().cuda()
    res = model(t)
    print(res.shape)