import torch
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MHA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(
            channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(
            channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(
            q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GFFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GFFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(
            channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(
            hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x

#####-----attention_to_1x1 used for Gray Scale Attention----#####


class attention_to_1x1(nn.Module):
    def __init__(self, channels):
        super(attention_to_1x1, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels*2, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(channels*2, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = torch.mean(x, -1)
        x = torch.mean(x, -1)
        x = torch.unsqueeze(x, -1)
        x = torch.unsqueeze(x, -1)
        xx = self.conv2(self.conv1(x))
        return xx


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MHA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GFFN(channels, expansion_factor)

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
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


def Phase_swap(x, y):
    fftn1 = torch.fft.fftn(x)
    fftn2 = torch.fft.fftn(y)
    out = torch.fft.ifftn(
        abs(fftn2)*torch.exp(1j*(fftn1.angle())))  # pHASE SWAPPING
    return out.real


class UIETPA(nn.Module):
    def __init__(self, num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], channels=[16, 32, 64, 128], num_refinement=4,
                 expansion_factor=2.66, ch=[16, 16, 32, 64]):
        super(UIETPA, self).__init__()
        self.sig = nn.Sigmoid()
        self.to_1x1 = nn.ModuleList(
            [attention_to_1x1(num_ch) for num_ch in ch])

        self.embed_conv_rgb = nn.Conv2d(
            3, channels[0], kernel_size=3, padding=1, bias=False)
        self.embed_conv_gray = nn.Conv2d(
            1, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
            zip(num_blocks, num_heads, channels)])
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch)
                                   for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch)
                                 for num_ch in list(reversed(channels))[:-1]])
        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        # the channel of last one is not change
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(
            channels[1], 3, kernel_size=3, padding=1, bias=False)

    def forward(self, RGB_input, Gray_input):

        ###-------encoder-------####

        fo_rgb = self.embed_conv_rgb(RGB_input)
        fo_gray = self.embed_conv_gray(Gray_input)
        out_enc_rgb1 = self.encoders[0](
            self.sig(self.to_1x1[0](fo_gray)*fo_rgb))
        out_enc_gray1 = self.encoders[0](fo_gray)
        out_enc_rgb2 = self.encoders[1](self.downs[0](
            self.sig(self.to_1x1[1](out_enc_gray1)*out_enc_rgb1)))
        out_enc_gray2 = self.encoders[1](self.downs[0](out_enc_gray1))
        out_enc_rgb3 = self.encoders[2](self.downs[1](
            self.sig(self.to_1x1[2](out_enc_gray2)*out_enc_rgb2)))
        out_enc_gray3 = self.encoders[2](self.downs[1](out_enc_gray2))
        out_enc_rgb4 = self.encoders[3](self.downs[2](
            self.sig(self.to_1x1[3](out_enc_gray3)*out_enc_rgb3)))

        ###-------Dencoder------####

        OUT1 = Phase_swap(out_enc_rgb3, self.ups[0](out_enc_rgb4))
        out_dec3 = self.decoders[0](self.reduces[0](
            torch.cat([self.ups[0](out_enc_rgb4), OUT1], dim=1)))
        OUT2 = Phase_swap(out_enc_rgb2, self.ups[1](out_dec3))
        out_dec2 = self.decoders[1](self.reduces[1](
            torch.cat([self.ups[1](out_dec3), OUT2], dim=1)))
        OUT3 = Phase_swap(out_enc_rgb1, self.ups[2](out_dec2))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), OUT3], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr)
        return out+RGB_input
