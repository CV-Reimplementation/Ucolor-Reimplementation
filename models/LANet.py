import torch
import torch.nn as nn


'''........Laplace operation.......'''
class Laplace(nn.Module):
    def __init__(self):
        super(Laplace,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=1,kernel_size=3,stride=1,padding=0,bias=False)
        nn.init.constant_(self.conv1.weight,1)
        nn.init.constant_(self.conv1.weight[0,0,1,1],-8)
        nn.init.constant_(self.conv1.weight[0,1,1,1],-8)
        nn.init.constant_(self.conv1.weight[0,2,1,1],-8)
      
    def forward(self,x1):
        edge_map=self.conv1(x1)
        return edge_map


class PALayer(nn.Module):
    '''........pixel attention(PA).......'''
    def __init__(self, channel):
        super(PALayer, self).__init__()
        
        self.PA = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(64),
            nn.Conv2d(channel // 8, 1, 3, padding=1, bias=True),
            # CxHxW -> 1xHxW
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.PA(x)
        return x * y

class CALayer(nn.Module):
    '''........Channel attention(CA).......'''
    def __init__(self, channel):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CA = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.CA(y)
        return x * y

class Block(nn.Module):
    '''........parallel attention module(PAM).......'''

    def __init__(self, dim, kernel_size):
        super(Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=(kernel_size // 2), bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=(kernel_size // 2), bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.conv1(x)
        res1 = self.calayer(res)
        res2 = self.palayer(res)
        res = res2 + res1
        res = self.conv2(res)
        res = res + x
        return res

class GS(nn.Module):
    '''........Group structure.......'''
    def __init__(self, dim, kernel_size, blocks):
        super(GS, self).__init__()
        modules = [Block(dim, kernel_size) for _ in range(blocks)]
        self.gs = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gs(x)
        return res

class Branch(nn.Module):
    '''......Branch......'''
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(Branch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.relu(x2)
        x4 = self.IN(x3)
        x5 = self.conv2(x4)

        return x1, x5

class LANet(nn.Module):
    '''......the structure of LANet......'''
    def __init__(self, gps = 3, blocks = 20, dim = 64, kernel_size = 3):
        super(LANet, self).__init__()
        self.gps = gps
        self.dim = dim
        self.kernel_size = kernel_size
        
        self.laplace = Laplace()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        assert self.gps == 3

        self.g1 = GS(self.dim, kernel_size, blocks=blocks)
        self.g2 = GS(self.dim, kernel_size, blocks=blocks)
        self.g3 = GS(self.dim, kernel_size, blocks=blocks)

        self.brabch_3 = Branch(in_channels = 3, out_channels = self.dim, kernel_size = 3)
        self.brabch_5 = Branch(in_channels=3, out_channels=self.dim, kernel_size=5)
        self.brabch_7 = Branch(in_channels=3, out_channels=self.dim, kernel_size=7)

        self.fusion = nn.Sequential(*[
            nn.Conv2d(self.dim * self.gps, self.dim // 8, 3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(self.dim),
            nn.Conv2d(self.dim // 8, self.gps, 1, padding = 0, bias=True),
            nn.Sigmoid()
        ])

        self.Final =  nn.Conv2d(self.dim, 3, 1, padding=0, bias=True)
        
    def forward(self, x):

        '''.....three branch.......'''
        x11, x1 = self.brabch_3(x)
        x22, x2 = self.brabch_5(x)
        x33, x3 = self.brabch_7(x)

        '''......Multiscale Fusion......'''
        w = self.fusion(torch.cat([x1, x2, x3], dim=1))
        w = torch.split(w, 1, dim = 1)
        x4 = w[0] * x1 + w[1] * x2 + w[2] * x3

        res1 = self.g1(x4)     #GS(1)
        '''......Adaptive learning Module......'''
        x5 = self.avg_pool(x4)
        res1= x5 * res1 + x33
        
        res2 = self.g2(res1)   #GS(2)
        '''......Adaptive learning Module......'''
        x6 = self.avg_pool(res1)
        res2 = x6 * res2 + x22
        
        res3 = self.g3(res2)    #GS(3)
        '''......Adaptive learning Module......'''
        x7 = self.avg_pool(res2)
        res3 = x7 * res3 + x11
       
        out = self.Final(res3)
        # Laplace operation
        edge_map = self.laplace(out)
        
        return out, edge_map 
    
if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256).cuda()
    model = LANet().cuda()
    res, _ = model(t)
    print(res.shape)