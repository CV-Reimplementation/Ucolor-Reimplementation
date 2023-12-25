import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.fft as fft


class AGSSF(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(AGSSF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size(), padding=(self.kernel_size() - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels)/self.gamma)+ self.b/self.gamma))
        out = k if k % 2 else k+1
        return out

    def forward(self, x):

        # x1=inv_mag(x)
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)


        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)


        # Multi-scale information fusion
        y = self.sigmoid(y)


        return x * y.expand_as(x)


class SFCA(nn.Module):
    def __init__(self, channels, relu_slope=0.2, gamma=2):
        super(SFCA, self).__init__()
        self.identity1 = nn.Conv2d(channels, channels, 1)
        self.identity2 = nn.Conv2d(channels, channels, 1)
        self.conv_1 = nn.Conv2d(channels, 2*channels, kernel_size=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope)
        self.conv_2 = nn.Conv2d(2*channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope)

        self.conv_f1 = nn.Conv2d(channels, 2*channels, kernel_size=1)
        self.conv_f2 = nn.Conv2d(2*channels, channels, kernel_size=1)
        self.con2X1 = nn.Conv2d(2*channels, channels, kernel_size=1)
        self.agssf = AGSSF(channels)

    def forward(self, x):
        out = self.conv_1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([out_1, out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        # print(self.identity1(x).shape, out.shape)
        out += self.identity1(x)

        x_fft = fft.fftn(x, dim=(-2, -1)).real
        x_fft = F.gelu(self.conv_f1(x_fft))
        x_fft = self.conv_f2(x_fft)
        x_reconstructed = fft.ifftn(x_fft, dim=(-2, -1)).real
        x_reconstructed += self.identity2(x)

        f_out = self.con2X1(torch.cat([out, x_reconstructed], dim=1))

        return self.agssf(f_out)


class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
    
        #frequency
    
        self.kv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.q1X1_1 = nn.Conv2d(channels, channels , kernel_size=1, bias=False)
        self.q1X1_2 = nn.Conv2d(channels, channels , kernel_size=1, bias=False)
        self.kv_conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2, bias=False)
        self.project_outf = nn.Conv2d(channels, channels, kernel_size=1, bias=False)



    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
  
        # frequency

        x_fft = fft.fftn(x, dim=(-2, -1)).real
        x_fft1=self.q1X1_1(x_fft)
        x_fft2=F.gelu(x_fft1)
        x_fft3=self.q1X1_2(x_fft2)
        qf=fft.ifftn(x_fft3,dim=(-2, -1)).real

        
        kf, vf = self.kv_conv(self.kv(out)).chunk(2, dim=1)
        qf = qf.reshape(b, self.num_heads, -1, h * w)
        kf = kf.reshape(b, self.num_heads, -1, h * w)
        vf = vf.reshape(b, self.num_heads, -1, h * w)
        qf, kf = F.normalize(qf, dim=-1), F.normalize(kf, dim=-1)
        attnf = torch.softmax(torch.matmul(qf, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        outf = self.project_outf(torch.matmul(attn, vf).reshape(b, -1, h, w))
        return outf


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


    

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)



class UpSample(nn.Module):
    def __init__(self, channels,channel_red):
        super(UpSample, self).__init__()
        
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        if channel_red:
            self.post = nn.Conv2d(channels, channels//2, 1, 1, 0)
            
        else:
            self.post = nn.Conv2d(channels, channels, 1, 1, 0)
        
    
    def forward(self, x):
        N, C, H, W = x.shape
        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)
        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)
        amp_fuse = torch.tile(Mag, (2, 2))
        pha_fuse = torch.tile(Pha, (2, 2))
        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)
        output = torch.fft.ifft2(out)
        output = torch.abs(output)  
        return self.post(output)
    
  
class UpSample1(nn.Module):
    def __init__(self, channels):
        super(UpSample1, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)  


class UpS(nn.Module):
    def __init__(self, channels):
        super(UpS, self).__init__()
        self.Fups=UpSample(channels,True)
        self.Sups=UpSample1(channels)
        self.reduce=nn.Conv2d(channels, channels // 2, kernel_size=1,bias=False)
    
    def forward(self, x):
        out=torch.cat([self.Fups(x),self.Sups(x)],dim=1)
        # print(out.shape)
        return self.reduce(out) 


class Model(nn.Module):
    def __init__(self, num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], channels=[16, 32, 64, 128], num_refinement=4,
                 expansion_factor=2.66, ch=[64,32,16,64]):
        super(Model, self).__init__()
        
        self.attention = nn.ModuleList([SFCA(num_ch) for num_ch in ch]) 
        self.embed_conv_rgb = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])

        self.down1 = DownSample(channels[0])
        self.down2 = DownSample(channels[1])
        self.down3 = DownSample(channels[2])
        self.ups_1=UpS(128)
        self.ups_2=UpS(64)
        self.ups_3=UpS(32)
        self.ups_4=UpS(3)
        
        self.ups1 = UpSample1(32)
        self.reduces2 = nn.Conv2d(64, 32, kernel_size=1, bias=False)    
        self.reduces1=nn.Conv2d(128, 64, kernel_size=1, bias=False)

        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))

        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor) for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(8, 3, kernel_size=3, padding=1, bias=False)
        self.output1= nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False)
                                 
        self.ups2 = UpSample1(16)
        self.outputl=nn.Conv2d(32, 8, kernel_size=3, padding=1, bias=False)
                                 
    def forward(self, RGB_input):
        ###-------encoder for RGB-------####
        fo_rgb = self.embed_conv_rgb(RGB_input)
        out_enc_rgb1 = self.encoders[0](fo_rgb)
        out_enc_rgb2 = self.encoders[1](self.down1(out_enc_rgb1))
        # print(out_enc_rgb2.shape)

        out_enc_rgb3 = self.encoders[2](self.down2(out_enc_rgb2))
        # print(out_enc_rgb3.shape)
        out_enc_rgb4 = self.encoders[3](self.down3(out_enc_rgb3))
        # print(out_enc_rgb4.shape)

        ###-------Dencoder------###
        out_dec3 = self.decoders[0](self.reduces1(torch.cat([(self.ups_1(out_enc_rgb4)), out_enc_rgb3], dim=1)))
        # print(out_dec3.shape)
        out_dec2 = self.decoders[1](self.reduces2(torch.cat([self.ups_2(out_dec3),out_enc_rgb2], dim=1)))
        # print(out_dec2.shape)     
        fd = self.decoders[2](torch.cat([self.ups_3(out_dec2),out_enc_rgb1], dim=1))
        # print(fd.shape)   
        # print('lasst',fd_FP.shape)
        fr = self.refinement(fd)
        return self.output(self.outputl(fr))


if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256).cuda()
    model = Model().cuda()
    res = model(t)
    print(res.shape)