import torch
import torch.nn as nn

def swish(x):
    return x * x.sigmoid()

def hard_sigmoid(x, inplace=False):
    return nn.ReLU6(inplace=inplace)(x + 3) / 6

def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)

class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, inplace=self.inplace)

class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)

def _make_divisible(v, divisor=8, min_value=None):  ## 将通道数变成8的整数倍
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Conv2d(oup, _make_divisible(inp // reduction), 1, 1, 0,),
                nn.ReLU(),
                nn.Conv2d(_make_divisible(inp // reduction), oup, 1, 1, 0),
                HardSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ConvBlock1(nn.Module):
    def __init__(self):
        super(ConvBlock1, self).__init__()
        self.DW = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, groups=16, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(16)
        self.HS = HardSwish()
        self.PW = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(32)

    def forward(self, x):
        a = self.HS(self.BN(self.DW(x)))
        a = self.HS(self.BNN(self.PW(a)))
        return a

class ConvBlock2(nn.Module):
    def __init__(self):
        super(ConvBlock2, self).__init__()
        self.DW = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, groups=32, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(32)
        self.HS = HardSwish()
        self.PW = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(64)

    def forward(self, x):
        a = self.HS(self.BN(self.DW(x)))
        a = self.HS(self.BNN(self.PW(a)))
        return a

class ConvBlock3(nn.Module):
    def __init__(self):
        super(ConvBlock3, self).__init__()
        self.DW = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, groups=64, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(64)
        self.HS = HardSwish()
        self.PW = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(32)

    def forward(self, x):
        a = self.HS(self.BN(self.DW(x)))
        a = self.HS(self.BNN(self.PW(a)))
        return a

class ConvBlock4(nn.Module):
    def __init__(self):
        super(ConvBlock4, self).__init__()
        self.DW = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, stride=1, groups=80, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(80)
        self.HS = HardSwish()
        self.PW = nn.Conv2d(in_channels=80, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(32)
        self.SE = SELayer(80, 80)

    def forward(self, x):

        a = self.HS(self.BN(self.DW(x)))
        a = self.SE(a)
        a = self.HS(self.BNN(self.PW(a)))
        return a

class LiteEnhanceNet(nn.Module):
    def __init__(self):
        super(LiteEnhanceNet, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)  ## 第一层卷积
        self.output = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = ConvBlock1()
        self.block2 = ConvBlock2()
        self.block3 = ConvBlock3()
        self.block4 = ConvBlock4()

    def forward(self, x):
        x = self.input(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        # x2 = torch.cat((x, x2), 1)
        x3 = self.block3(x2)
        x3 = torch.cat((x, x1, x3), 1)
        x4 = self.block4(x3)
        out = self.output(x4)
        return out
    

if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256).cuda()
    model = LiteEnhanceNet().cuda()
    res = model(t)
    print(res.shape)