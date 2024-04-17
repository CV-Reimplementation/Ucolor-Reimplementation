import math

import torch.nn as nn
from kornia.color import rgb_to_hsv
import torch
from ptflops import get_model_complexity_info


class UIA(nn.Module):
    def __init__(self, channels, ks):
        super(UIA, self).__init__()
        self._c_avg = nn.AdaptiveAvgPool2d((1, 1))
        self._c_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self._c_sig = nn.Sigmoid()
        self._h_avg = nn.AdaptiveAvgPool2d((1, None))
        self._h_conv = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self._w_avg = nn.AdaptiveAvgPool2d((None, 1))
        self._w_conv = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self._hw_conv = nn.Conv2d(channels, channels, ks, padding=ks // 2, padding_mode='reflect',
                                  groups=channels, bias=False)
        self._chw_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self._chw_sig = nn.Sigmoid()

    def forward(self, x):
        c_map = self._c_conv(self._c_avg(x))
        c_weight = self._c_sig(c_map)
        h_map = self._h_conv(self._h_avg(x))
        w_map = self._w_conv(self._w_avg(x))
        hw_map = self._hw_conv(w_map @ h_map)
        chw_map = self._chw_conv(c_weight * hw_map)
        chw_weight = self._chw_sig(chw_map)
        return chw_weight * x


class NormGate(nn.Module):
    def __init__(self, channels, ks, norm=nn.InstanceNorm2d):
        super(NormGate, self).__init__()
        self._norm_branch = nn.Sequential(
            norm(channels),
            nn.Conv2d(channels, channels, ks, padding=ks // 2, padding_mode='reflect', bias=False)
        )
        self._sig_branch = nn.Sequential(
            nn.Conv2d(channels, channels, ks, padding=ks // 2, padding_mode='reflect', bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        norm = self._norm_branch(x)
        sig = self._sig_branch(x)
        return norm * sig


class UCB(nn.Module):
    def __init__(self, channels, ks):
        super(UCB, self).__init__()
        self._body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, padding=ks // 2,
                      padding_mode='reflect', bias=False),
            NormGate(channels, ks),
            UIA(channels, ks)
        )

    def forward(self, x):
        y = self._body(x)
        return y + x


class PWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(PWConv, self).__init__()
        self._body = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                               padding=kernel_size // 2, padding_mode='reflect', bias=bias)

    def forward(self, x):
        return self._body(x)


class GlobalColorCompensationNet(nn.Module):
    def __init__(self, channel_scale, kernel_size):
        super(GlobalColorCompensationNet, self).__init__()
        self._body = nn.Sequential(
            PWConv(3, channel_scale, kernel_size),
            UCB(channel_scale, kernel_size),
            UCB(channel_scale, kernel_size),
            UCB(channel_scale, kernel_size),
            PWConv(channel_scale, 3, kernel_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self._body(x)
        return y


class CLCC(nn.Module):
    def __init__(self, channel_scale, main_ks, gcc_ks):
        super(CLCC, self).__init__()
        self._color_branch = GlobalColorCompensationNet(channel_scale, gcc_ks)
        self._in_conv = nn.Sequential(
            PWConv(3, channel_scale, main_ks),
            UIA(channel_scale, main_ks)
        )
        self._group1 = nn.Sequential(
            *[UCB(channel_scale, main_ks) for _ in range(4)]
        )
        self._group2 = nn.Sequential(
            *[UCB(channel_scale, main_ks) for _ in range(4)]
        )
        self._group3 = nn.Sequential(
            *[UCB(channel_scale, main_ks) for _ in range(4)]
        )
        self._group1_adaptation = nn.Sequential(
            PWConv(3, channel_scale, main_ks),
            UCB(channel_scale, main_ks)
        )
        self._group2_adaptation = nn.Sequential(
            PWConv(3, channel_scale, main_ks),
            UCB(channel_scale, main_ks)
        )
        self._group3_adaptation = nn.Sequential(
            PWConv(3, channel_scale, main_ks),
            UCB(channel_scale, main_ks)
        )
        self._out_conv = nn.Sequential(
            PWConv(channel_scale, 3, main_ks),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            # elif isinstance(m, nn.InstanceNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()


    def forward(self, x):
        color_comp = 1 - x
        color_comp_map = self._color_branch(color_comp)
        in_feat = self._in_conv(x)
        group1_out = self._group1(in_feat)
        group1_comp_out = group1_out + self._group1_adaptation(color_comp_map * color_comp)
        group2_out = self._group2(group1_comp_out)
        group2_comp_out = group2_out + self._group2_adaptation(color_comp_map * color_comp)
        group3_out = self._group3(group2_comp_out)
        group3_comp_out = group3_out + self._group3_adaptation(color_comp_map * color_comp)
        out = self._out_conv(group3_comp_out)
        return out


if __name__ == '__main__':
    import torch
    x = torch.randn((2, 3, 256, 256))
    model = CLCC(64, 3, 3)
    macs, params = get_model_complexity_info(model, (3, 256, 256), verbose=False, print_per_layer_stat=False)
    print('MACS: ' + str(macs))
    print('Params: ' + str(params))
    # model = GlobalColorCompensationNet(64)
    y = model(x)
    print(y.shape)