import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import numpy as np
from einops import rearrange



##########################################################################
def window_partition(x, window_size: int,h,w):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    pad_l = pad_t = 0
    pad_r = (window_size - w % window_size) % window_size
    pad_b = (window_size - h % window_size) % window_size
    x = F.pad(x, [ pad_l, pad_r, pad_t, pad_b])
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    pad_l = pad_t = 0
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    H = H+pad_b
    W = W+pad_r
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, -1, H // window_size, W // window_size, window_size, window_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
    windows = F.pad(x, [pad_l, -pad_r, pad_t, -pad_b])
    return windows
##########################################################################


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## FFN
class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*3)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.relu(x1) * x2
        x = self.project_out(x)
        return x


class SWPSA(nn.Module):
    def __init__(self, dim, window_size, shift_size, bias):
        super(SWPSA, self).__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,groups=dim * 3, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1,bias=bias)
        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1,bias=bias)

        self.qkv_conv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv1 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,groups=dim * 3, bias=bias)

    def window_partitions(self,x, window_size: int):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size(M)

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def create_mask(self, x):

        n,c,H,W = x.shape
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self.window_partitions(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        shortcut = x
        b, c, h, w = x.shape

        x = window_partition(x,self.window_size,h,w)

        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q.transpose(-2,-1) @ k)/self.window_size
        attn = attn.softmax(dim=-1)
        out = (v @ attn )
        out = rearrange(out, 'b c (h w) -> b c h w', h=int(self.window_size),
                        w=int(self.window_size))
        out = self.project_out(out)
        out = window_reverse(out,self.window_size,h,w)

        shift = torch.roll(out,shifts=(-self.shift_size,-self.shift_size),dims=(2,3))
        shift_window = window_partition(shift,self.window_size,h,w)
        qkv = self.qkv_dwconv1(self.qkv_conv1(shift_window))
        q, k,v  = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q.transpose(-2,-1) @ k)/self.window_size
        mask = self.create_mask(shortcut)
        attn = attn.view(b,-1,self.window_size*self.window_size,self.window_size*self.window_size) + mask.unsqueeze(0)
        attn = attn.view(-1,self.window_size*self.window_size,self.window_size*self.window_size)
        attn = attn.softmax(dim=-1)

        out = (v @ attn)

        out = rearrange(out, 'b c (h w) -> b c h w', h=int(self.window_size),
                        w=int(self.window_size))

        out = self.project_out1(out)
        out = window_reverse(out,self.window_size,h,w)
        out = torch.roll(out,shifts=(self.shift_size,self.shift_size),dims=(2,3))

        return out

class SWPSATransformerBlock(nn.Module):
    def __init__(self, dim, window_size, shift_size, bias):
        super(SWPSATransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = SWPSA(dim, window_size, shift_size, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim,bias)

    def forward(self, x):

        y = self.attn(self.norm1(x))
        diffY = x.size()[2] - y.size()[2]
        diffX = x.size()[3] - y.size()[3]

        y = F.pad(y, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = x+y
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads

        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1,bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv_conv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))/np.sqrt(int(c/self.num_heads))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class CATransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CATransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = ChannelAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim,bias)

    def forward(self, x):
        y = self.attn(self.norm1(x))
        diffY = x.size()[2] - y.size()[2]
        diffX = x.size()[3] - y.size()[3]

        y = F.pad(y, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = x + y
        x = x + self.ffn(self.norm2(x))

        return x
class LatentPixelAttention(nn.Module):
    def __init__(self, dim, bias):
        super(LatentPixelAttention, self).__init__()
        self.qkv = nn.Conv2d(dim , dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim *  3, dim *  3, kernel_size=3, stride=1, padding=1,
                                            groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1,bias=bias)


    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q.transpose(-2,-1) @ k)
        attn = attn.softmax(dim=-1)

        out = (v @ attn )

        out = rearrange(out, 'b c (h w) -> b c h w', h=int(h),
                        w=int(w))

        out = self.project_out(out)

        return out

class LatentPixelTransformerBlock(nn.Module):
    def __init__(self, dim, bias):
        super(LatentPixelTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = LatentPixelAttention(dim, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim,bias)

    def forward(self, x):

        y = self.attn(self.norm1(x))
        diffY = x.size()[2] - y.size()[2]
        diffX = x.size()[3] - y.size()[3]

        y = F.pad(y, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = x+y
        x = x + self.ffn(self.norm2(x))


        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, bias):
        super(TransformerBlock, self).__init__()

        self.globa = CATransformerBlock(dim, num_heads, bias)

        self.pixel = SWPSATransformerBlock(dim,window_size,shift_size,bias)

        self.alpha = nn.Parameter(torch.ones(1)/2)

    def forward(self, x):

        x = self.alpha * self.pixel(x) + (1-self.alpha)*self.globa(x)

        return x

class LatentTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, bias):
        super(LatentTransformerBlock, self).__init__()

        self.csa = CATransformerBlock(dim, num_heads, bias)

        self.pixel = LatentPixelTransformerBlock(dim,bias)

        self.belta = nn.Parameter(torch.ones(1)/2)

    def forward(self, x):

        x = self.belta * self.pixel(x) + (1-self.belta)*self.csa(x)

        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)

def cat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)

    return x


##########################################################################
class UDAformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=36,
                 num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=2,
                 heads=[2, 2, 2,2],
                 bias=False,
                 window_size = 8,
                 shift_size = 3
                 ):

        super(UDAformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], bias=bias,window_size=window_size,shift_size=shift_size) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            CATransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], bias=bias) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            CATransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], bias=bias) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            LatentTransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], bias=bias,window_size=window_size,shift_size=shift_size) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            CATransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], bias=bias) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            CATransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], bias=bias) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], bias=bias,window_size=window_size,shift_size=shift_size) for i in range(num_blocks[0])])

        self.reduce_chan_ref = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.refinement = nn.Sequential(*[
            CATransformerBlock(dim=int(dim), num_heads=heads[0], bias=bias) for i in range(num_refinement_blocks)])



        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=1,bias=bias)




    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = cat(inp_dec_level3, out_enc_level3)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = cat(inp_dec_level2, out_enc_level2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)


        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = cat(inp_dec_level1, out_enc_level1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        ref_out = self.refinement(out_dec_level1)

        out = self.output(ref_out) + inp_img



        return out


if __name__ == '__main__':
    data = torch.randn([1, 3, 128, 128])
    model = UDAformer()
    out = model(data)
    print(out.shape)
    