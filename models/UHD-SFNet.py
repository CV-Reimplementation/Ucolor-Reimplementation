import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
from functools import partial
from einops.layers.torch import Rearrange
from torchvision import transforms


pair = lambda x: x if isinstance(x, tuple) else (x, x)

class VAE(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(VAE, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = 2 * kernel_size * kernel_size

        self.fc1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
        self.fc2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)

 
    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(x))
        return h1, h2 # mu, log_var
 
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return nn.Sigmoid()(z)

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)
        self.vae = VAE(in_channels, kernel_size=kernel_size[0])
    def forward(self, x):

        R_offset = self.offset_conv(x) * self.vae(x)
        
        
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=R_offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x



class Head(nn.Module):
    def __init__(self, size1=256, size2=256):
        super(Head, self).__init__()
        


        self.c1 = DeformableConv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.c2 = DeformableConv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1)
        
        self.h = nn.AdaptiveAvgPool2d((size1, size2))


    def forward(self, x):
        
        h = self.h(x)
        y = self.c1(h)
        y = nn.ReLU()(y)
        y = self.c2(y) + h
        return y


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        #self.con1 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.con1 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        return self.con1(self.fn(self.norm(x)) + x)

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.PReLU(),
        dense(inner_dim, dim),
        nn.PReLU()

    )

def MLPMixer(*, image_size, channels, patch_size, dim, depth, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.PReLU(),
        nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.PReLU(),
        Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size,
                  h = int(image_h/patch_size), w = int(image_w/patch_size)),
    )

class ConvBlock(nn.Module):
    def __init__(self, inc , outc, kernel_size=3, padding=1, stride=1, 
                 use_bias=True, activation=nn.ReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, 
                              padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap): 
        device = bilateral_grid.get_device()

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)], indexing='ij') # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) # norm to [0,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) # norm to [0,1] NxHxWx1
        hg, wg = hg*2-1, wg*2-1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, align_corners=True)
        return coeff.squeeze(2)

class GuideNN(nn.Module):
    def __init__(self, bn=True):
        super(GuideNN, self).__init__()

        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=bn)
        self.conv2 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Tanh)

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)

        return output

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
    
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            #nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
            #          groups=1, bias=True),
            Rearrange('b c h w -> b h w c'),
            nn.Linear(dw_channel // 2, dw_channel // 2),
            Rearrange('b h w c -> b c h w')
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 4, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.sg_b = nn.Parameter(torch.zeros((1, ffn_channel // 2, 1, 1)), requires_grad=True)
    def forward(self, inp):

        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.sg(x)
        x = x1 * self.sca(x1)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x+x1 * self.sg_b)
        x = self.conv5(x)

        x = self.dropout2(x)

        return inp + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=2, enc_blk_nums=[2, 2, 2, 20], dec_blk_nums=[2, 2, 2, 2]):
    #def __init__(self, img_channel=3, width=32, middle_blk_num=20, enc_blk_nums=[2, 2, 2, 2], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class Pyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = Head(256, 256)
        self.h2 = Head(128, 128)
        self.gs = transforms.GaussianBlur(kernel_size=33)
        self.gs1 = transforms.GaussianBlur(kernel_size=101)
        self.f_g = MLPMixer(
            image_size = 256,
            #image_size = 128,
            channels = 3,
            patch_size = 16,
            dim = 768,
            depth = 6
        )
        self.f1 = MLPMixer(
            #image_size = 256,
            image_size = 128,
            channels = 3,
            patch_size = 16,
            dim = 768,
            depth = 12
        )
        self.f2 = MLPMixer(
            #image_size = 256,
            image_size = 128,
            channels = 3,
            patch_size = 16,
            dim = 768,
            depth = 12
        )
        self.g = GuideNN()
        self.slice = Slice()
        self.squeeze_3d = nn.Conv3d(24, 3, kernel_size=3, stride=1, padding=1)
        self.u_n = NAFNet()


    def forward(self, x):
        
        g = self.g(x)

        x1 = self.h1(x)

      
        x_g = self.gs(x1)
        u_f = self.u_n(x_g)
        u_f = F.interpolate(u_f, (x.shape[2], x.shape[3]),
                            mode='bilinear', align_corners=True)
        x_f = torch.fft.fft2(x)
        real = x_f.real
        imag = x_f.imag

        feature_real = F.interpolate(real, (128, 128),
                            mode='bilinear', align_corners=True)

        feature_imag = F.interpolate(imag, (128, 128),
                            mode='bilinear', align_corners=True)

        x1 = self.f1(feature_real).view(-1, 12, 16, 16, 16)
        x2 = self.f2(feature_imag).view(-1, 12, 16, 16, 16)
        coeff = self.squeeze_3d(torch.cat((x1, x2), dim=1))
        
        attention_map = self.slice(coeff, g)
 
        return attention_map * x
    

if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256).cuda()
    model = Pyramid().cuda()
    res = model(t)
    print(res.shape)
    