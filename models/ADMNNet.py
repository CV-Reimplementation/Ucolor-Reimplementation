import torch
import torch.nn as nn

def similarity_matrix(x):
    B, C, H, W = x.size()
    scale = ((H * W) ** -0.5)
    bmat1 = x.flatten(start_dim=2)
    bmat2 = bmat1.transpose(1, 2)
    bmat3 = torch.bmm(bmat1, bmat2, out=None)
    bmat3 = bmat3 * scale
    similarity = nn.Softmax(dim=-1)(bmat3)
    return similarity


class PoolBlock(nn.Module):
    def __init__(self, in_channels, stride=2):
        super(PoolBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.prelu1 = nn.PReLU(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.prelu2 = nn.PReLU(in_channels)

    def forward(self, x):
        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.prelu2(self.bn2(self.conv2(x)))
        return x


class convLayer(nn.Module):
    """
    "Depthwise conv + Pointwise conv"
    """

    def __init__(self, in_channels, out_channels, dilation=1, kernel_size=3, stride=1):
        super(convLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=dilation,
                               groups=in_channels, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.prelu1 = nn.PReLU(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.prelu2(self.bn2(self.conv2(x)))
        return x


class conv1_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv1_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu1 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.prelu1(self.bn1(self.conv1(x)))
        return x


class CA(nn.Module):
    def __init__(self, in_channels):
        super(CA, self).__init__()
        self.pool1 = PoolBlock(in_channels)

    def forward(self, x):
        B, C, H, W = x.size()
        pool1 = self.pool1(x)

        attention = similarity_matrix(pool1)
        x1 = x.flatten(start_dim=2).transpose(1, 2)
        out = torch.bmm(x1, attention, out=None)
        out = out.transpose(1, 2).view(B, C, H, W)

        return out


class Multiscale(nn.Module):
    def __init__(self, in_channels):
        super(Multiscale, self).__init__()
        self.conv1 = conv1_1(in_channels, in_channels)
        self.conv3 = convLayer(in_channels, in_channels)
        self.conv5 = convLayer(in_channels, in_channels, dilation=2)

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.conv3(x)

        x3 = self.conv5(x)

        return x1, x2, x3


class DFSM(nn.Module):
    def __init__(self, in_channels):
        super(DFSM, self).__init__()
        self.Multiscale = Multiscale(in_channels)

        self.pool1 = PoolBlock(in_channels)
        self.conv1 = conv1_1(in_channels, 1)

        self.pool2 = PoolBlock(in_channels)
        self.conv2 = conv1_1(in_channels, 1)

        self.pool3 = PoolBlock(in_channels)
        self.conv3 = conv1_1(in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.size()
        x1, x2, x3 = self.Multiscale(x)
        pool1 = self.conv1(self.pool1(x1))  # 1*1
        pool2 = self.conv2(self.pool2(x2))  # 3*3
        pool3 = self.conv3(self.pool3(x3))  # 5*5

        cat = torch.cat([pool1, pool2, pool3], dim=1)
        attention = similarity_matrix(cat)  # 3*3

        x1_1 = x1.view(B, -1, C, H, W).flatten(start_dim=2)
        x2_2 = x2.view(B, -1, C, H, W).flatten(start_dim=2)
        x3_3 = x3.view(B, -1, C, H, W).flatten(start_dim=2)

        bmat1 = torch.cat([x1_1, x2_2, x3_3], dim=1).transpose(1, 2)

        out = torch.bmm(bmat1, attention)
        out = out.transpose(1, 2)

        x1 = x1 + out[:, 0, :].view(B, C, H, W)
        x2 = x2 + out[:, 1, :].view(B, C, H, W)
        x3 = x3 + out[:, 2, :].view(B, C, H, W)

        return x1 + x2 + x3


class MCAM(nn.Module):
    def __init__(self, in_channels):
        super(MCAM, self).__init__()
        self.Multiscale = Multiscale(in_channels)

        self.ca1 = CA(in_channels)
        self.ca2 = CA(in_channels)
        self.ca3 = CA(in_channels)

    def forward(self, x):
        x1, x2, x3 = self.Multiscale(x)
        x_ca1 = self.ca1(x1)
        x_ca2 = self.ca2(x2)
        x_ca3 = self.ca3(x3)
        return x_ca1 + x_ca2 + x_ca3


class Block(nn.Module):
    def __init__(self, in_channels):
        super(Block, self).__init__()
        self.dfsm = DFSM(in_channels)
        self.mcam = MCAM(in_channels)

    def forward(self, x):
        x_dfsm = self.dfsm(x) + x
        x_mcam = self.mcam(x_dfsm) + x_dfsm

        return x_mcam + x


class ADMNNet(nn.Module):
    def __init__(self, in_channels=3, blocks_num=3, out_channels=16):
        super(ADMNNet, self).__init__()
        self.first = convLayer(in_channels, out_channels)
        self.blocks = nn.Sequential(*[Block(out_channels) for i in range(blocks_num)])
        self.conv2_final = nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        y = self.first(x)
        y = self.blocks(y)

        mask = self.tanh(self.conv2_final(y))

        out = torch.clamp(mask + x, -1, 1, out=None)
        return out


if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256)
    net = ADMNNet(3, 3)
    out = net(t)
    print(out.shape)
