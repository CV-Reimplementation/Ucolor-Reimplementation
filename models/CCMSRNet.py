import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import random
import time

def normalize_img(img):
    if torch.max(img) > 1 or torch.min(img) < 0:
        # img: b x c x h x w
        b, c, h, w = img.shape
        temp_img = img.view(b, c, h*w)
        im_max = torch.max(temp_img, dim=2)[0].view(b, c, 1)
        im_min = torch.min(temp_img, dim=2)[0].view(b, c, 1)

        temp_img = (temp_img - im_min) / (im_max - im_min + 1e-7)
        
        img = temp_img.view(b, c, h, w)
    
    return img


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down_Axial_onlyV(nn.Module):
    def __init__(self, in_channels, out_channels, key_dim, num_heads):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = nn.Sequential(          
            DoubleConv(in_channels, out_channels),
        )
        self.pool4trans = nn.AdaptiveAvgPool2d((16,16))
        self.attn = Sea_Attention_onlyV(dim = in_channels, key_dim=key_dim, num_heads=num_heads)
        self.transition = nn.Conv2d(in_channels,out_channels,1)

    def forward(self, x):
        x = self.pool(x)      
        x_conv = self.conv(x)
        b,c,h,w = x_conv.shape

        x_trans = self.pool4trans(x)
        x_trans = self.attn(x_trans)
        x_trans = self.transition(x_trans)
        x_trans = F.interpolate(x_trans,size=(h,w),mode='bilinear')
        return x_conv + x_trans

class Conv2d_BN(nn.Module):
    def __init__(self,in_channel,out_channel,ks=1,stride=1,pad=0,dilation=1,groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=ks, stride=stride, padding=pad, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channel)
    def forward(self,x):
        return self.bn(self.conv(x))


class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]), requires_grad=True)

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        return x
    

class Sea_Attention_onlyV(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=2,
                 activation=nn.ReLU,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads

        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1 )
        self.to_k = Conv2d_BN(dim, nh_kd, 1 )
        self.to_v = Conv2d_BN(dim, self.dh, 1 )
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim ))
        self.proj_encode_row = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh ))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_column = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh ))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        
        self.dwconv = Conv2d_BN(self.dh,  self.dh, ks=3, stride=1, pad=1, dilation=1,
                 groups=self.dh )
        self.act = activation()
        self.pwconv = Conv2d_BN(self.dh, dim, ks=1 )

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        

        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        
        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        
        ## squeeze column
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        
        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)
        xx = self.proj(xx)
        return xx

class Up_Axial_onlyV(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, key_dim, num_heads,bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        self.pool4trans = nn.AdaptiveAvgPool2d((16,16))
        self.attn = Sea_Attention_onlyV(dim=in_channels,key_dim=key_dim, num_heads=num_heads)
        self.transition = nn.Conv2d(in_channels,out_channels,1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x_conv = self.conv(x)
        b,c,h,w = x_conv.shape

        x_trans = self.pool4trans(x)
        x_trans = self.attn(x_trans)
        x_trans = self.transition(x_trans)
        x_trans = F.interpolate(x_trans,size=(h,w),mode='bilinear')
        return x_conv+x_trans

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class CCNet(nn.Module):
    def __init__(self, n_channels, n_classes,img_size, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        factor = 2 if bilinear else 1
        self.bilinear = bilinear

        #self.b = nn.Linear(1024//factor,n_channels)

        
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down_Axial_onlyV(16, 32,16,4)
        self.down2 = Down_Axial_onlyV(32, 64,16,4)
        self.down3 = Down_Axial_onlyV(64, 128,16,4)          
        self.down4 = Down_Axial_onlyV(128, 128,16,4)

        self.up1 = Up_Axial_onlyV(256,64,16,4)
        self.up2 = Up_Axial_onlyV(128,32,16,4)
        self.up3 = Up_Axial_onlyV(64,16,16,4)
        self.up4 = Up_Axial_onlyV(32,16,16,4)
        self.outc = OutConv(16,2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        ### color balance
        b,c,h,w=  x.shape
        I = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        maps = self.outc(x)
        maps = self.sigmoid(maps)
        r_map = maps[:,0,:,:]
        b_map = maps[:,1,:,:]
        R,G,B = I[:,0,:,:],I[:,1,:,:],I[:,2,:,:]

        R = R + r_map*(G-R)*(1-R)*G
        B = B + b_map*(G-B)*(1-B)*G
        
        R = R.unsqueeze(1)
        G = G.unsqueeze(1)
        B = B.unsqueeze(1)
        out = torch.cat([R,G,B],dim=1)
        x_cc = normalize_img(out)
        
        return x_cc

class UNetAxialFuser(nn.Module):
    def __init__(self, n_channels, n_classes,img_size, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        factor = 2 if bilinear else 1
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down_Axial_onlyV(64, 128,16,4)
        self.down2 = Down_Axial_onlyV(128, 256,16,4)
        self.down3 = Down_Axial_onlyV(256, 512,16,4)
        
        self.down4 = Down_Axial_onlyV(512, 1024 // factor,16,4)
        self.up1 = Up_Axial_onlyV(1024, 512 // factor, 16,4,bilinear)
        self.up2 = Up_Axial_onlyV(512, 256 // factor, 16,4,bilinear)
        self.up3 = Up_Axial_onlyV(256, 128 // factor,16,4,bilinear)
        self.up4 = Up_Axial_onlyV(128, 64,16,4, bilinear)

        self.outc = OutConv(64,n_classes)
    
    def forward(self, x):

        b,c,h,w = x.shape

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        ### F decoder

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = self.outc(x)
        return out

class CCMSRNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3,img_size=256, bilinear=True):
        super(CCMSRNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        factor = 2 if bilinear else 1
        self.bilinear = bilinear
        self.ccnet = CCNet(n_channels, n_classes,img_size, bilinear)
        self.fuser = UNetAxialFuser(9,3,img_size,bilinear)

    def forward(self, x):
        x_cc = self.ccnet(x)

        I = x_cc
        ssr1 = torch.log(I+1/255)*(1 - torch.log(TF.gaussian_blur(I+1/255,kernel_size=3)))
        ssr2 = torch.log(I+1/255)*(1 - torch.log(TF.gaussian_blur(I+1/255,kernel_size=7)))
        ssr3 = torch.log(I+1/255)*(1 - torch.log(TF.gaussian_blur(I+1/255,kernel_size=11)))
        msr_cat = torch.cat([ssr1,ssr2,ssr3],dim=1)
        msr_fuse = self.fuser(msr_cat)

        msr = normalize_img(msr_fuse)
        return msr
    
if __name__ == '__main__':
    inp = torch.randn(1, 3, 256, 256).cuda()
    model = CCMSRNet().cuda()
    res = model(inp)
    print(res.shape)