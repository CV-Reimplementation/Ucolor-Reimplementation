import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = torch.sigmoid_(self.fc(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)


def get_residue(tensor, r_dim=1):
    """
    return residue_channle (RGB)
    """
    # res_channel = []
    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]
    return res_channel


class convd(nn.Module):
    def __init__(self, inputchannel, outchannel, kernel_size, stride):
        super(convd, self).__init__()
        self.relu = nn.ReLU()
        self.padding = nn.ReflectionPad2d(kernel_size//2)
        self.conv = nn.Conv2d(inputchannel, outchannel, kernel_size, stride)
        self.ins = nn.InstanceNorm2d(outchannel, affine=True)

    def forward(self, x):
        x = self.conv(self.padding(x))
        # x= self.ins(x)
        x = self.relu(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Upsample, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.relu(out)
        out = F.interpolate(out, y.size()[2:])
        return out


class RB(nn.Module):
    def __init__(self, n_feats, nm='in'):
        super(RB, self).__init__()
        module_body = []
        for i in range(2):
            module_body.append(
                nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
            module_body.append(nn.ReLU())
        self.module_body = nn.Sequential(*module_body)
        self.relu = nn.ReLU()
        self.se = SELayer(n_feats, 1)

    def forward(self, x):
        res = self.module_body(x)
        res = self.se(res)
        res += x
        return res


class RIR(nn.Module):
    def __init__(self, n_feats, n_blocks, nm='in'):
        super(RIR, self).__init__()
        module_body = [
            RB(n_feats) for _ in range(n_blocks)
        ]
        module_body.append(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
        self.module_body = nn.Sequential(*module_body)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.module_body(x)
        res += x
        return self.relu(res)


class res_ch(nn.Module):
    def __init__(self, n_feats, blocks=2):
        super(res_ch, self).__init__()
        self.conv_init1 = convd(3, n_feats//2, 3, 1)
        self.conv_init2 = convd(n_feats//2, n_feats, 3, 1)
        self.extra = RIR(n_feats, n_blocks=blocks)

    def forward(self, x):
        x = self.conv_init2(self.conv_init1(x))
        x = self.extra(x)
        return x


class Fuse(nn.Module):
    def __init__(self, inchannel=64, outchannel=64):
        super(Fuse, self).__init__()
        self.up = Upsample(inchannel, outchannel, 3, 2)
        self.conv = convd(outchannel, outchannel, 3, 1)
        self.rb = RB(outchannel)
        self.relu = nn.ReLU()

    def forward(self, x, y):

        x = self.up(x, y)
        # x = F.interpolate(x, y.size()[2:])
        # y1 = torch.cat((x, y), dim=1)
        y = x+y

        # y = self.pf(y1) + y

        return self.relu(self.rb(y))


class Prior_Sp(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim=32):
        super(Prior_Sp, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        # self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        # self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x, prior):

        x_q = self.query_conv(x)
        prior_k = self.key_conv(prior)
        energy = x_q * prior_k
        attention = torch.sigmoid_(energy)
        # print(attention.size(),x.size())
        attention_x = x * attention
        attention_p = prior * attention

        x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

        p_gamma = self.gamma2(torch.cat((prior, attention_p), dim=1))
        prior_out = prior * p_gamma[:, [0], :, :] + \
            attention_p * p_gamma[:, [1], :, :]

        return x_out, prior_out


# Basic modules
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4,
                      stride=2, padding=1, bias=bias)
    return layer


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), stride=stride, bias=bias)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            if i == 0:
                m.append(conv(n_feats, 64, kernel_size, bias=bias))
            else:
                m.append(conv(64, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


##########################################################################
# Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = torch.sigmoid_(self.conv_du(y))
        return x * y


##########################################################################
# Compute inter-stage features
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x1 = x1 + x
        return x1, img


class mergeblock(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, subspace_dim=16):
        super(mergeblock, self).__init__()
        self.conv_block = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.num_subspace = subspace_dim
        self.subnet = conv(n_feat * 2, self.num_subspace,
                           kernel_size, bias=bias)

    def forward(self, x, bridge):
        out = torch.cat([x, bridge], 1)
        b_, c_, h_, w_ = bridge.shape
        sub = self.subnet(out)
        V_t = sub.view(b_, self.num_subspace, h_*w_)
        V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis=2, keepdims=True))
        V = V_t.permute(0, 2, 1)
        mat = torch.matmul(V_t, V)
        mat_inv = torch.inverse(mat)
        project_mat = torch.matmul(mat_inv, V_t)
        bridge_ = bridge.view(b_, c_, h_*w_)
        project_feature = torch.matmul(project_mat, bridge_.permute(0, 2, 1))
        bridge = torch.matmul(V, project_feature).permute(
            0, 2, 1).view(b_, c_, h_, w_)
        out = torch.cat([x, bridge], 1)
        out = self.conv_block(out)
        return out+x


##########################################################################
# U-Net
class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff, depth=5):
        super(Encoder, self).__init__()
        self.body = nn.ModuleList()  # []
        self.depth = depth
        for i in range(depth-1):
            self.body.append(UNetConvBlock(in_size=n_feat+scale_unetfeats*i, out_size=n_feat +
                             scale_unetfeats*(i+1), downsample=True, relu_slope=0.2, use_csff=csff, use_HIN=True))
        self.body.append(UNetConvBlock(in_size=n_feat+scale_unetfeats*(depth-1), out_size=n_feat +
                         scale_unetfeats*(depth-1), downsample=False, relu_slope=0.2, use_csff=csff, use_HIN=True))

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        res = []
        if encoder_outs is not None and decoder_outs is not None:
            for i, down in enumerate(self.body):
                if (i+1) < self.depth:
                    x, x_up = down(x, encoder_outs[i], decoder_outs[-i-1])
                    res.append(x_up)
                else:
                    x = down(x)
        else:
            for i, down in enumerate(self.body):
                if (i+1) < self.depth:
                    x, x_up = down(x)
                    res.append(x_up)
                else:
                    x = down(x)
        return res, x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(
            in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(
            out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(in_size, out_size, 3, 1, 1)
            self.phi = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.gamma = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            skip_ = F.leaky_relu(self.csff_enc(
                enc) + self.csff_dec(dec), 0.1, inplace=True)
            out = out * torch.sigmoid_(self.phi(skip_)) + \
                self.gamma(skip_) + out
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(
            out_size*2, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=5):
        super(Decoder, self).__init__()

        self.body = nn.ModuleList()
        self.skip_conv = nn.ModuleList()  # []
        for i in range(depth-1):
            self.body.append(UNetUpBlock(in_size=n_feat+scale_unetfeats*(depth-i-1),
                             out_size=n_feat+scale_unetfeats*(depth-i-2), relu_slope=0.2))
            self.skip_conv.append(nn.Conv2d(
                n_feat+scale_unetfeats*(depth-i-1), n_feat+scale_unetfeats*(depth-i-2), 3, 1, 1))

    def forward(self, x, bridges):
        res = []
        for i, up in enumerate(self.body):
            x = up(x, self.skip_conv[i](bridges[-i-1]))
            res.append(x)

        return res


##########################################################################
# ---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


##########################################################################
# DGUNet_plus
class DGUNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4, bias=False, depth=5):
        super(DGUNet, self).__init__()
        # Extract Shallow Features
        n_feats = 3
        blocks = 2
        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat5 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat6 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat7 = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        # Gradient Descent Module (GDM)
        self.phi_0 = ResBlock(default_conv, 3, 3)
        self.phit_0 = ResBlock(default_conv, 3, 3)
        self.phi_5 = ResBlock(default_conv, 3, 3)
        self.phit_5 = ResBlock(default_conv, 3, 3)
        self.phi_6 = ResBlock(default_conv, 3, 3)
        self.phit_6 = ResBlock(default_conv, 3, 3)
        self.r0 = nn.Parameter(torch.Tensor([0.5]))
        self.r5 = nn.Parameter(torch.Tensor([0.5]))
        self.r6 = nn.Parameter(torch.Tensor([0.5]))

        # Informative Proximal Mapping Module (IPMM)
        self.stage1_encoder = Encoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4, csff=False)
        self.stage1_decoder = Decoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4)

        self.stage6_encoder = Encoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4, csff=True)
        self.stage6_decoder = Decoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.merge56 = mergeblock(n_feat, 3, True)
        self.sam67 = SAM(n_feat, kernel_size=1, bias=bias)
        self.merge67 = mergeblock(n_feat, 3, True)

        self.tail = conv(n_feat, 3, kernel_size, bias=bias)
        # prior guided
        self.res_extra1 = res_ch(n_feats, blocks)
        self.output1 = nn.Conv2d(n_feats, 3, 3, 1, padding=1)

    def forward(self, img):
        # prior giuded
        res_x = get_residue(img)
        x1 = self.res_extra1(torch.cat((res_x, res_x, res_x), dim=1))
        prior1 = self.output1(x1)
        # -------------------------------------------
        # -------------- Stage 1---------------------
        # -------------------------------------------
        # GDM

        phixsy_1 = self.phi_0(img) - img + prior1
        x1_img = img - self.r0*self.phit_0(phixsy_1)
        # PMM
        x1 = self.shallow_feat1(x1_img)
        feat1, feat_fin1 = self.stage1_encoder(x1)
        res1 = self.stage1_decoder(feat_fin1, feat1)
        x2_samfeats, stage1_img = self.sam12(res1[-1], x1_img)

        # GDM
        phixsy_6 = self.phi_5(stage1_img) - img + prior1
        x6_img = stage1_img - self.r5*self.phit_5(phixsy_6)
        # PMM
        x6 = self.shallow_feat6(x6_img)
        x6_cat = self.merge56(x6, x2_samfeats)
        feat6, feat_fin6 = self.stage6_encoder(x6_cat, feat1, res1)
        res6 = self.stage6_decoder(feat_fin6, feat6)
        x7_samfeats, stage6_img = self.sam67(res6[-1], x6_img)

        # -------------------------------------------
        # -------------- Stage 7---------------------
        # -------------------------------------------
        # GDM
        phixsy_7 = self.phi_6(stage6_img) - img + prior1
        x7_img = stage6_img - self.r6*self.phit_6(phixsy_7)
        # PMM
        x7 = self.shallow_feat7(x7_img)
        x7_cat = self.merge67(x7, x7_samfeats)
        stage7_img = self.tail(x7_cat) + img

        # return [stage7_img, stage6_img, stage1_img]
        return stage7_img


if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256).cuda()
    model = DGUNet().cuda()
    out = model(t)
    print(out.shape)
