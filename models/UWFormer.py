from pytorch_wavelets import DWTForward, DWTInverse
import torch
import torch.nn as nn
import torch.nn.functional as F


class LAM_Module_v2(nn.Module):
    """ Layer attention module"""

    def __init__(self, in_dim, bias=True):
        super(LAM_Module_v2, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d(self.chanel_in, self.chanel_in * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in * 3, self.chanel_in * 3, kernel_size=3, stride=1, padding=1,
                                    groups=self.chanel_in * 3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()

        x_input = x.view(m_batchsize, N * C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1 + x
        out = out.view(m_batchsize, -1, height, width)
        return out


def gauss_kernel(channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel


class LapPyramidConv(nn.Module):
    def __init__(self, num_high=4):
        super(LapPyramidConv, self).__init__()

        self.num_high = num_high
        self.kernel = gauss_kernel()

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel.to(img.device), groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image
    

class LPAttention(nn.Module):
    def __init__(self, depth=2, num_dims=3, bias=True):
        super().__init__()
        self.lap_pyramid = LapPyramidConv(depth)
        self.relu = nn.PReLU()
        # k conv
        self.conv1 = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.conv3 = nn.Conv2d(num_dims * 6, num_dims * 3, kernel_size=3, padding=1, bias=bias)
        self.conv4 = nn.Conv2d(num_dims * 6, num_dims * 3, kernel_size=3, padding=1, bias=bias)
        # v conv
        self.conv1_ = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.conv2_ = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.conv3_ = nn.Conv2d(num_dims * 6, num_dims * 3, kernel_size=3, padding=1, bias=bias)
        self.conv4_ = nn.Conv2d(num_dims * 6, num_dims * 3, kernel_size=3, padding=1, bias=bias)

        self.conv5 = nn.Conv2d(num_dims * 3, num_dims, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        q = self.conv2(self.conv1(x))
        q = self.relu(q)
        pyr_inp = self.lap_pyramid.pyramid_decom(img=x)

        k1 = self.conv2(self.conv1(pyr_inp[-1]))
        k2 = self.conv2(self.conv1(pyr_inp[-2]))
        k3 = self.conv2(self.conv1(pyr_inp[-3]))
        # k1->k2
        k1 = nn.functional.interpolate(k1, size=(pyr_inp[-2].shape[2], pyr_inp[-2].shape[3]))
        k1k2 = torch.cat([k1, k2], dim=1)
        k1k2 = self.conv3(k1k2)
        # k1k2(k2)->k3
        k1k2 = nn.functional.interpolate(k1k2, size=(pyr_inp[-3].shape[2], pyr_inp[-3].shape[3]))
        k = torch.cat([k1k2, k3], dim=1)
        k = self.conv4(k)
        k = self.relu(k)

        v1 = self.conv2_(self.conv1_(pyr_inp[-1]))
        v2 = self.conv2_(self.conv1_(pyr_inp[-2]))
        v3 = self.conv2_(self.conv1_(pyr_inp[-3]))
        v1 = nn.functional.interpolate(v1, size=(pyr_inp[-2].shape[2], pyr_inp[-2].shape[3]))
        v1v2 = torch.cat([v1, v2], dim=1)
        v1v2 = self.conv3_(v1v2)
        v1v2 = nn.functional.interpolate(v1v2, size=(pyr_inp[-3].shape[2], pyr_inp[-3].shape[3]))
        v = torch.cat([v1v2, v3], dim=1)
        v = self.conv4_(v)
        v = self.relu(v)

        qk = q @ k.transpose(2, 3)
        qkv = qk @ v
        qkv = self.conv5(qkv)
        qkv = self.relu(qkv)
        return qkv


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


# Dual Gated Feed-Forward Networ
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x2) * x1 + F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# in_channels=c, LayerNorm2d(c)
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=True):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm2d(dim)
        self.lpattn = LPAttention()
        # self.lpattn = NextAttentionZ(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.lpattn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x
    


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
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
                nn.Conv2d(chan, 2 * chan, 2, 2)
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
    

class LPViT(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=3,
                 num_blocks=[1, 2, 4, 8],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 attention=True
                 ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for _ in
            range(num_blocks[0])])

        self.encoder_2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_blocks[0])])

        self.encoder_3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_blocks[0])])

        self.layer_fussion = LAM_Module_v2(in_dim=int(dim * 3))
        self.conv_fuss = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)

        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_blocks[0])])

        self.trans_low = NAFNet()

        self.coefficient_1_0 = nn.Parameter(torch.ones((2, int(int(dim)))), requires_grad=attention)

        self.refinement_1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_refinement_blocks)])
        self.refinement_2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_refinement_blocks)])
        self.refinement_3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias) for
            _ in range(num_refinement_blocks)])

        self.layer_fussion_2 = LAM_Module_v2(in_dim=int(dim * 3))
        self.conv_fuss_2 = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1, bias=bias)

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp):
        inp_enc_encoder1 = self.patch_embed(inp)
        # print(inp_enc_encoder1.shape)
        out_enc_encoder1 = self.encoder_1(inp_enc_encoder1)
        # print(out_enc_encoder1.shape)
        out_enc_encoder2 = self.encoder_2(out_enc_encoder1)
        # print(out_enc_encoder2.shape)
        out_enc_encoder3 = self.encoder_3(out_enc_encoder2)
        # print(out_enc_encoder3.shape)

        inp_fusion_123 = torch.cat(
            [out_enc_encoder1.unsqueeze(1), out_enc_encoder2.unsqueeze(1), out_enc_encoder3.unsqueeze(1)], dim=1)

        out_fusion_123 = self.layer_fussion(inp_fusion_123)
        out_fusion_123 = self.conv_fuss(out_fusion_123)

        out_enc = self.trans_low(out_fusion_123)

        out_fusion_123 = self.latent(out_fusion_123)

        out = self.coefficient_1_0[0, :][None, :, None, None] * out_fusion_123 + self.coefficient_1_0[1, :][None, :,
                                                                                 None, None] * out_enc

        out_1 = self.refinement_1(out)
        out_2 = self.refinement_2(out_1)
        out_3 = self.refinement_3(out_2)

        inp_fusion = torch.cat([out_1.unsqueeze(1), out_2.unsqueeze(1), out_3.unsqueeze(1)], dim=1)
        out_fusion_123 = self.layer_fussion_2(inp_fusion)
        out = self.conv_fuss_2(out_fusion_123)
        result = self.output(out)

        return result


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class FourierUnit(torch.nn.Module):
    """Implements Fourier Unit block.

    Applies FFT to tensor and performs convolution in spectral domain.
    After that return to time domain with Inverse FFT.

    Attributes:
        inter_conv: conv-bn-relu block that performs conv in spectral domain

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            fu_kernel: int = 1,
            padding_type: str = "reflect",
            fft_norm: str = "ortho",
            use_only_freq: bool = True,
            norm_layer=nn.BatchNorm2d,
            bias: bool = True,
    ):
        super().__init__()
        self.fft_norm = fft_norm
        self.use_only_freq = use_only_freq

        self.inter_conv = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                out_channels * 2,
                kernel_size=fu_kernel,
                stride=1,
                padding=get_padding(fu_kernel),
                padding_mode=padding_type,
                bias=bias,
            ),
            norm_layer(out_channels * 2),
            nn.ReLU(True),
        )

    def forward(self, x):
        batch_size, ch, freq_dim, embed_dim = x.size()

        dims_to_fft = (-2,) if self.use_only_freq else (-2, -1)
        recover_length = (freq_dim,) if self.use_only_freq else (
            freq_dim, embed_dim)

        fft_representation = torch.fft.rfftn(
            x, dim=dims_to_fft, norm=self.fft_norm)

        # (B, Ch, 2, FFT_freq, FFT_embed)
        fft_representation = torch.stack(
            (fft_representation.real, fft_representation.imag), dim=2
        )  # .view(batch_size, ch * 2, -1, embed_dim)

        ffted_dims = fft_representation.size()[-2:]
        fft_representation = fft_representation.view(
            (
                batch_size,
                ch * 2,
            )
            + ffted_dims
        )

        fft_representation = (
            self.inter_conv(fft_representation)
            .view(
                (
                    batch_size,
                    ch,
                    2,
                )
                + ffted_dims
            )
            .permute(0, 1, 3, 4, 2)
        )

        fft_representation = torch.complex(
            fft_representation[..., 0], fft_representation[..., 1]
        )

        reconstructed_x = torch.fft.irfftn(
            fft_representation, dim=dims_to_fft, s=recover_length, norm=self.fft_norm
        )

        assert reconstructed_x.size() == x.size()

        return reconstructed_x


class SpectralTransform(torch.nn.Module):
    """Implements Spectrals Transform block.

    Residual Block containing Fourier Unit with convolutions before and after.

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            fu_kernel: int = 1,
            padding_type: str = "reflect",
            fft_norm: str = "ortho",
            use_only_freq: bool = True,
            norm_layer=nn.BatchNorm2d,
            bias: bool = False,
    ):
        super().__init__()
        halved_out_ch = out_channels // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, halved_out_ch,
                      kernel_size=1, stride=1, bias=bias),
            norm_layer(halved_out_ch),
            nn.ReLU(True),
        )

        self.fu = FourierUnit(
            halved_out_ch,
            halved_out_ch,
            fu_kernel=fu_kernel,
            use_only_freq=use_only_freq,
            fft_norm=fft_norm,
            padding_type=padding_type,
            norm_layer=norm_layer,
        )

        self.conv2 = nn.Conv2d(
            halved_out_ch, out_channels, kernel_size=1, stride=1, bias=bias
        )

    def forward(self, x):
        residual = self.conv1(x)
        x = self.fu(residual)
        x += residual
        x = self.conv2(x)

        return x


class FastFourierConvolution(torch.nn.Module):
    """Implements FFC block.

    Divides Tensor in two branches: local and global. Local branch performs
    convolutions and global branch applies Spectral Transform layer.
    After performing transforms in local and global branches outputs are passed through BatchNorm + ReLU
    and eventually concatenated. Based on proportion of input and output global channels if the number is equal
    to zero respective blocks are replaced by Identity Transform.
    For clarity refer to original paper.

    Attributes:
        local_in_channels: # input channels for l2l and l2g convs
        local_out_channels: # output channels for l2l and g2l convs
        global_in_channels: # input channels for g2l and g2g convs
        global_out_channels: # output_channels for l2g and g2g convs
        l2l_layer: local to local Convolution
        l2g_layer: local to global Convolution
        g2l_layer: global to local Convolution
        g2g_layer: global to global Spectral Transform

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            alpha_in: float = 0.5,
            alpha_out: float = 0.5,
            kernel_size: int = 3,
            padding_type: str = "reflect",
            fu_kernel: int = 1,
            fft_norm: str = "ortho",
            bias: bool = True,
            norm_layer=nn.BatchNorm2d,
            activation=nn.ReLU(True),
            use_only_freq: bool = True,
    ):
        """Inits FFC module.

        Args:
            in_channels: total channels of tensor before dividing into local and global
            alpha_in:
                proportion of global channels as input
            alpha_out:
                proportion of global channels as output
            use_only_freq:
                controls dimensionality of fft in Fourier Unit. If false uses 2D fft in Fourier Unit affecting both
                frequency and time dimensions, otherwise applies 1D FFT only to frequency dimension

        """
        super().__init__()
        self.global_in_channels = int(in_channels * alpha_in)
        self.local_in_channels = in_channels - self.global_in_channels
        self.global_out_channels = int(out_channels * alpha_out)
        self.local_out_channels = out_channels - self.global_out_channels

        padding = get_padding(kernel_size)

        tmp_module = self._get_module_on_true_predicate(
            self.local_in_channels > 0 and self.local_out_channels > 0,
            nn.Conv2d,
            nn.Identity,
        )
        self.l2l_layer = tmp_module(
            self.local_in_channels,
            self.local_out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_type,
            bias=bias,
        )

        tmp_module = self._get_module_on_true_predicate(
            self.local_in_channels > 0 and self.global_out_channels > 0,
            nn.Conv2d,
            nn.Identity,
        )
        self.l2g_layer = tmp_module(
            self.local_in_channels,
            self.global_out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_type,
            bias=bias,
        )

        tmp_module = self._get_module_on_true_predicate(
            self.global_in_channels > 0 and self.local_out_channels > 0,
            nn.Conv2d,
            nn.Identity,
        )
        self.g2l_layer = tmp_module(
            self.global_in_channels,
            self.local_out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_type,
            bias=bias,
        )

        tmp_module = self._get_module_on_true_predicate(
            self.global_in_channels > 0 and self.global_out_channels > 0,
            SpectralTransform,
            nn.Identity,
        )
        self.g2g_layer = tmp_module(
            self.global_in_channels,
            self.global_out_channels,
            fu_kernel=fu_kernel,
            fft_norm=fft_norm,
            padding_type=padding_type,
            bias=bias,
            norm_layer=norm_layer,
            use_only_freq=use_only_freq,
        )

        self.local_bn_relu = (
            nn.Sequential(norm_layer(self.local_out_channels), activation)
            if self.local_out_channels != 0
            else nn.Identity()
        )

        self.global_bn_relu = (
            nn.Sequential(norm_layer(self.global_out_channels), activation)
            if self.global_out_channels != 0
            else nn.Identity()
        )

    @staticmethod
    def _get_module_on_true_predicate(
            condition: bool, true_module=nn.Identity, false_module=nn.Identity
    ):
        if condition:
            return true_module
        else:
            return false_module

    def forward(self, x):

        #  chunk into local and global channels
        x_l, x_g = (
            x[:, : self.local_in_channels, ...],
            x[:, self.local_in_channels:, ...],
        )
        x_l = 0 if x_l.size()[1] == 0 else x_l
        x_g = 0 if x_g.size()[1] == 0 else x_g

        out_local, out_global = torch.Tensor(0).to(x.device), torch.Tensor(0).to(
            x.device
        )

        if self.local_out_channels != 0:
            out_local = self.l2l_layer(x_l) + self.g2l_layer(x_g)
            out_local = self.local_bn_relu(out_local)

        if self.global_out_channels != 0:
            out_global = self.l2g_layer(x_l) + self.g2g_layer(x_g)
            out_global = self.global_bn_relu(out_global)

        #  (B, out_ch, F, T)
        output = torch.cat((out_local, out_global), dim=1)

        return output


class FFCResNetBlock(torch.nn.Module):
    """Implements Residual FFC block.

    Contains two FFC blocks with residual connection.

    Wraps around FFC arguments.

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            alpha_in: float = 0.5,
            alpha_out: float = 0.5,
            kernel_size: int = 3,
            padding_type: str = "reflect",
            bias: bool = True,
            fu_kernel: int = 1,
            fft_norm: str = "ortho",
            use_only_freq: bool = True,
            norm_layer=nn.BatchNorm2d,
            activation=nn.ReLU(True),
    ):
        super().__init__()
        self.ffc1 = FastFourierConvolution(
            in_channels,
            out_channels,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            kernel_size=kernel_size,
            padding_type=padding_type,
            fu_kernel=fu_kernel,
            fft_norm=fft_norm,
            use_only_freq=use_only_freq,
            bias=bias,
            norm_layer=norm_layer,
            activation=activation,
        )

        # self.ffc2 = FastFourierConvolution(
        #     in_channels,
        #     out_channels,
        #     alpha_in=alpha_in,
        #     alpha_out=alpha_out,
        #     kernel_size=kernel_size,
        #     padding_type=padding_type,
        #     fu_kernel=fu_kernel,
        #     fft_norm=fft_norm,
        #     use_only_freq=use_only_freq,
        #     bias=bias,
        #     norm_layer=norm_layer,
        #     activation=activation,
        # )

    def forward(self, x):
        out = self.ffc1(x)
        # out = self.ffc2(out)
        return x + out


class FFCNet(nn.Module):

    def __init__(self):
        super(FFCNet, self).__init__()

        self.conv1 = nn.Conv2d(9, 64, kernel_size=9,
                               padding=4, padding_mode='reflect', stride=1)
        self.relu = nn.PReLU()

        self.resBlock = self._makeLayer_(FFCResNetBlock, 64, 64, 9)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.PReLU()

        self.convPos1 = nn.Conv2d(
            64, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.reluPos1 = nn.PReLU()

        self.convPos2 = nn.Conv2d(
            256, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.reluPos2 = nn.PReLU()

        self.finConv = nn.Conv2d(64, 9, kernel_size=1, stride=1)

    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        layers = []
        layers.append(block(inChannals, outChannals))

        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # print("conv1", x.shape)
        x = self.relu(x)
        residual = x

        out = self.resBlock(x)
        # print("res", out.shape)

        out = self.conv2(out)
        # print(out.shape)
        out = self.bn2(out)
        # print(out.shape)
        out += residual

        out = self.convPos1(out)
        # print(out.shape)
        # print(out.shape)
        out = self.reluPos1(out)
        # print(out.shape)

        out = self.convPos2(out)
        # print(out.shape)
        out = self.reluPos2(out)
        out = self.finConv(out)
        # print(out.shape)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Trans_low(nn.Module):
    def __init__(self, num_residual_blocks=5):
        super(Trans_low, self).__init__()

        model = [nn.Conv2d(9, 16, 3, padding=1),
                 nn.InstanceNorm2d(16),
                 nn.LeakyReLU(),
                 nn.Conv2d(16, 64, 3, padding=1),
                 nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(64)]

        model += [nn.Conv2d(64, 16, 3, padding=1),
                  nn.LeakyReLU(),
                  nn.Conv2d(16, 9, 3, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        out = torch.tanh(out)
        return out


class UWFormer(nn.Module):
    def __init__(self):
        super(UWFormer, self).__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.idwt = DWTInverse(mode='zero', wave='haar')
        self.ll_layer_module = LPViT()
        self.h_layer = FFCNet()
        self.criterion_l1 = torch.nn.SmoothL1Loss()

    def forward(self, inp):
        inp_ll, inp_hf = self.dwt(inp)

        inp_hl = inp_hf[0][:, :, 0, :, :]
        inp_lh = inp_hf[0][:, :, 1, :, :]
        inp_hh = inp_hf[0][:, :, 2, :, :]

        inp_ll_hat = self.ll_layer_module(inp_ll)
        inp_H = torch.cat((inp_hl, inp_lh, inp_hh), dim=1)
        # print(inp_H.shape)

        out_H = self.h_layer(inp_H)
        out_hl = out_H[:, 0:3, :, :]
        out_lh = out_H[:, 3:6, :, :]
        out_hh = out_H[:, 6:9, :, :]

        recon_hl = out_hl.unsqueeze(2)
        recon_lh = out_lh.unsqueeze(2)
        recon_hh = out_hh.unsqueeze(2)

        recon_hf = [torch.cat((recon_hl, recon_lh, recon_hh), dim=2)]

        result = self.idwt((inp_ll_hat, recon_hf))

        return result


if __name__ == '__main__':
    model = UWFormer()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)