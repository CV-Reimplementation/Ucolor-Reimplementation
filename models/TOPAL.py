import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=0, bias=True, activation='prelu', norm='batch', groups=1):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias, groups=groups)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm='batch'):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class Decoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm='batch', mode='iter1'):
        super(Decoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.down_convs.append(
                ConvBlock(num_filter*(2**i), num_filter*(2**(i+1)), kernel_size, stride, padding, bias, activation, norm='batch')
            )
            self.up_convs.append(
                DeconvBlock(num_filter*(2**(i+1)), num_filter*(2**i), kernel_size, stride, padding, bias, activation, norm='batch')
            )

    def forward(self, ft_h, ft_l_list):
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_h_list = []
            for i in range(len(ft_l_list)):
                ft_h_list.append(ft_h)
                ft_h = self.down_convs[self.num_ft- len(ft_l_list) + i](ft_h)

            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft_fusion = self.up_convs[self.num_ft-i-1](ft_fusion - ft_l_list[i]) + ft_h_list[len(ft_l_list)-i-1]

        if self.mode == 'iter2':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(i+1):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[len(ft_l_list) - i - 1]
                for j in range(i+1):
                    # print(j)
                    ft = self.up_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_h
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion

class Encoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm='batch', mode='iter1'):
        super(Encoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for i in range(self.num_ft):
            self.up_convs.append(
                DeconvBlock(num_filter//(2**i), num_filter//(2**(i+1)), kernel_size, stride, padding, bias, activation, norm='batch')
            )
            self.down_convs.append(
                ConvBlock(num_filter//(2**(i+1)), num_filter//(2**i), kernel_size, stride, padding, bias, activation, norm='batch')
            )

    def forward(self, ft_l, ft_h_list):
        if self.mode == 'iter1' or self.mode == 'conv':
            ft_l_list = []
            for i in range(len(ft_h_list)):
                ft_l_list.append(ft_l)
                ft_l = self.up_convs[self.num_ft- len(ft_h_list) + i](ft_l)

            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft_fusion = self.down_convs[self.num_ft-i-1](ft_fusion - ft_h_list[i]) + ft_l_list[len(ft_h_list)-i-1]

        if self.mode == 'iter2':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    # print(j)
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter3':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(i+1):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[len(ft_h_list) - i - 1]
                for j in range(i+1):
                    # print(j)
                    ft = self.down_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_l
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    # print(j)
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion

class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = ConvBlock(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, activation='relu', bias=False)

  def forward(self, x):
    out = self.conv(x)
    out = torch.cat((x, out), 1)
    return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate, scale = 1.0):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    self.scale = scale
    modules = []
    for i in range(nDenselayer):
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = ConvBlock(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out) * self.scale
    out = out + x
    return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = ConvBlock(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

class MSBDN(nn.Module):
    def __init__(self, res_blocks=18):
        super(MSBDN, self).__init__()

        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
        self.dense0 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )

        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.conv1 = RDB(16, 4, 16)
        self.fusion1 = Encoder_MDCBlock1(16, 2, mode='iter2')
        self.dense1 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv2 = RDB(32, 4, 32)
        self.fusion2 = Encoder_MDCBlock1(32, 3, mode='iter2')
        self.dense2 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.conv3 = RDB(64, 4, 64)
        self.fusion3 = Encoder_MDCBlock1(64, 4, mode='iter2')
        self.dense3 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.conv4 = RDB(128, 4, 128)
        self.fusion4 = Encoder_MDCBlock1(128, 5, mode='iter2')

        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(256))

        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.conv_4 = RDB(64, 4, 64)
        self.fusion_4 = Decoder_MDCBlock1(64, 2, mode='iter2')

        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_3 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.conv_3 = RDB(32, 4, 32)
        self.fusion_3 = Decoder_MDCBlock1(32, 3, mode='iter2')

        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_2 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )
        self.conv_2 = RDB(16, 4, 16)
        self.fusion_2 = Decoder_MDCBlock1(16, 4, mode='iter2')

        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
        self.dense_1 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )
        self.conv_1 = RDB(8, 4, 8)
        self.fusion_1 = Decoder_MDCBlock1(8, 5, mode='iter2')

        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)


    def forward(self, x):
        res1x = self.conv_input(x)
        res1x_1, res1x_2 = res1x.split([(res1x.size()[1] // 2), (res1x.size()[1] // 2)], dim=1)
        feature_mem = [res1x_1]
        x = self.dense0(res1x) + res1x

        res2x = self.conv2x(x)
        res2x_1, res2x_2 = res2x.split([(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1)
        res2x_1 = self.fusion1(res2x_1, feature_mem)
        res2x_2 = self.conv1(res2x_2)
        feature_mem.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)
        res2x =self.dense1(res2x) + res2x

        res4x =self.conv4x(res2x)
        res4x_1, res4x_2 = res4x.split([(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1)
        res4x_1 = self.fusion2(res4x_1, feature_mem)
        res4x_2 = self.conv2(res4x_2)
        feature_mem.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)
        res4x = self.dense2(res4x) + res4x

        res8x = self.conv8x(res4x)
        res8x_1, res8x_2 = res8x.split([(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1)
        res8x_1 = self.fusion3(res8x_1, feature_mem)
        res8x_2 = self.conv3(res8x_2)
        feature_mem.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)
        res8x = self.dense3(res8x) + res8x

        res16x = self.conv16x(res8x)
        res16x_1, res16x_2 = res16x.split([(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1)
        res16x_1 = self.fusion4(res16x_1, feature_mem)
        res16x_2 = self.conv4(res16x_2)
        res16x = torch.cat((res16x_1, res16x_2), dim=1)

        res_dehaze = res16x
        in_ft = res16x*2
        res16x = self.dehaze(in_ft) + in_ft - res_dehaze
        res16x_1, res16x_2 = res16x.split([(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1)
        feature_mem_up = [res16x_1]

        res16x = self.convd16x(res16x)
        res16x = F.interpolate(res16x, res8x.size()[2:], mode='bilinear')
        res8x = torch.add(res16x, res8x)
        res8x = self.dense_4(res8x) + res8x - res16x
        res8x_1, res8x_2 = res8x.split([(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1)
        res8x_1 = self.fusion_4(res8x_1, feature_mem_up)
        res8x_2 = self.conv_4(res8x_2)
        feature_mem_up.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)

        res8x = self.convd8x(res8x)
        res8x = F.interpolate(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)
        res4x = self.dense_3(res4x) + res4x - res8x
        res4x_1, res4x_2 = res4x.split([(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1)
        res4x_1 = self.fusion_3(res4x_1, feature_mem_up)
        res4x_2 = self.conv_3(res4x_2)
        feature_mem_up.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)


        res4x = self.convd4x(res4x)
        res4x = F.interpolate(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)
        res2x = self.dense_2(res2x) + res2x - res4x
        res2x_1, res2x_2 = res2x.split([(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1)
        res2x_1 = self.fusion_2(res2x_1, feature_mem_up)
        res2x_2 = self.conv_2(res2x_2)
        feature_mem_up.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)

        res2x = self.convd2x(res2x)
        res2x = F.interpolate(res2x, x.size()[2:], mode='bilinear')
        x = torch.add(res2x, x)
        x = self.dense_1(x) + x - res2x
        x_1, x_2 = x.split([(x.size()[1] // 2), (x.size()[1] // 2)], dim=1)
        x_1 = self.fusion_1(x_1, feature_mem_up)
        x_2 = self.conv_1(x_2)
        x = torch.cat((x_1, x_2), dim=1)

        x = self.conv_output(x)

        return x
    
class DoubleConvBlock(nn.Module):
    """double conv layers block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, padding=1, activation='relu'),
            ConvBlock(out_channels, out_channels, kernel_size=3, padding=1, activation='relu'),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscale block: maxpool -> double conv block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class BridgeDown(nn.Module):
    """Downscale bottleneck block: maxpool -> conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, kernel_size=3, padding=1, activation='relu'),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class BridgeUP(nn.Module):
    """Downscale bottleneck block: conv -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_up = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3, padding=1, activation='relu'),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv_up(x)



class UpBlock(nn.Module):
    """Upscale block: double conv block -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels * 2, in_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)



    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return torch.relu(self.up(x))


class OutputBlock(nn.Module):
    """Output block: double conv block -> output conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_conv = nn.Sequential(
            DoubleConvBlock(in_channels * 2, in_channels),
            ConvBlock(in_channels, out_channels, kernel_size=1))

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.out_conv(x)


class deepWBnet(nn.Module):
    def __init__(self):
        super(deepWBnet, self).__init__()
        self.n_channels = 3
        self.encoder_inc = DoubleConvBlock(self.n_channels, 24)
        self.encoder_down1 = DownBlock(24, 48)
        self.encoder_down2 = DownBlock(48, 96)
        self.encoder_down3 = DownBlock(96, 192)
        self.encoder_bridge_down = BridgeDown(192, 384)
        self.decoder_bridge_up = BridgeUP(384, 192)
        self.decoder_up1 = UpBlock(192, 96)
        self.decoder_up2 = UpBlock(96, 48)
        self.decoder_up3 = UpBlock(48, 24)
        self.decoder_out = OutputBlock(24, self.n_channels)


    def forward(self, x):
        x1 = self.encoder_inc(x)
        x2 = self.encoder_down1(x1)
        x3 = self.encoder_down2(x2)
        x4 = self.encoder_down3(x3)
        x5 = self.encoder_bridge_down(x4)
        x = self.decoder_bridge_up(x5)
        x = self.decoder_up1(x, x4)
        x = self.decoder_up2(x, x3)
        x = self.decoder_up3(x, x2)
        out = self.decoder_out(x, x1)
        return out


class RDB(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB, self).__init__()
        Cin = inChannels
        G = growRate

        self.conv1 = ConvBlock(Cin, G, kSize, padding=(kSize -1 )//2, stride=1, activation='lrelu')
        self.conv2 = ConvBlock(Cin + G, G, kSize, padding=(kSize -1 )//2, stride=1, activation='lrelu')
        self.conv3 = ConvBlock(Cin + 2 * G, G, kSize, padding=(kSize -1 )//2, stride=1, activation='lrelu')
        self.conv4 = ConvBlock(Cin + 3 * G, G, kSize, padding=(kSize - 1) // 2, stride=1, activation='lrelu')
        self.conv5 = ConvBlock(Cin + 4 * G, G, kSize, padding=(kSize - 1) // 2, stride=1, activation='lrelu')
        self.conv6 = ConvBlock(Cin + 5 * G, Cin, kSize, padding=(kSize - 1) // 2, stride=1, activation='lrelu')

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.interpolate(x1, x.shape[2:])

        x2 = self.conv2(torch.cat((x, x1), 1))
        x2 = F.interpolate(x2, x.shape[2:])

        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x3 = F.interpolate(x3, x.shape[2:])

        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x4 = F.interpolate(x4, x.shape[2:])

        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = F.interpolate(x5, x.shape[2:])

        x6 = self.conv6(torch.cat((x, x1, x2, x3, x4, x5), 1))
        x6 = F.interpolate(x6, x.shape[2:])

        return 0.1 * x6 + x

class WRDB(nn.Module):
    def __init__(self, num_features, growRate, kSize=3):
        super(WRDB, self).__init__()
        self.RDB1 = RDB(num_features, growRate, kSize)
        self.RDB2 = RDB(num_features, growRate, kSize)
        self.RDB3 = RDB(num_features, growRate, kSize)

        self.conv0= ConvBlock(num_features, num_features, kSize, padding=(kSize -1 )//2, stride=1, activation='lrelu')
        self.conv1_1 = ConvBlock(in_channels=num_features, out_channels=num_features,
                               kernel_size=1, padding=0, stride=1, groups=num_features)
        self.conv1_2 = ConvBlock(in_channels=num_features, out_channels=num_features,
                                 kernel_size=1, padding=0, stride=1, groups=num_features)
        self.conv1_3 = ConvBlock(in_channels=num_features, out_channels=num_features,
                                 kernel_size=1, padding=0, stride=1, groups=num_features)

        self.conv2_1 = ConvBlock(in_channels=num_features, out_channels=num_features,
                               kernel_size=1, padding=0, stride=1, groups=num_features)
        self.conv2_2 = ConvBlock(in_channels=num_features, out_channels=num_features,
                                 kernel_size=1, padding=0, stride=1, groups=num_features)

        self.conv3_1 = ConvBlock(in_channels=num_features, out_channels=num_features,
                               kernel_size=1, padding=0, stride=1, groups=num_features)
        
    def forward(self, x):
        x1_0 = self.conv0(x)
        x1_1 = self.conv1_1(x1_0)
        x2_0 = self.RDB1(x1_0)
        x2 = x2_0 + x1_1
        x2_1 = self.conv2_1(x2)
        x1_2 = self.conv1_2(x1_0)
        x3_0 = self.RDB2(x2)
        x3 = x2_1 + x1_2 + x3_0
        x1_3 = self.conv1_3(x1_0)
        x2_2 = self.conv2_2(x2)
        x3_1 = self.conv3_1(x3)
        x4 = self.RDB3(x3)
        out = x1_3 + x2_2 + x3_1 + x4
        out = 0.1 * out + x
        return out

class Upsampler(nn.Module):
    def __init__(self, num_features, out_channels, kSize, scale):
        super(Upsampler, self).__init__()
        # Up-sampling net
        if scale == 2 or scale == 3:
            self.UPNet = nn.Sequential(*[
                ConvBlock(num_features, num_features * scale * scale, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(scale),
                ConvBlock(num_features, out_channels, kSize, padding=(kSize - 1) // 2, stride=1)
            ])
        elif scale == 4:
            self.UPNet = nn.Sequential(*[
                ConvBlock(num_features, num_features * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                ConvBlock(num_features, num_features * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                ConvBlock(num_features, out_channels, kSize, padding=(kSize - 1) // 2, stride=1)
            ])
        elif scale == 8:
            self.UPNet = nn.Sequential(*[
            ConvBlock(num_features, num_features * 4, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            ConvBlock(num_features, num_features * 4, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            ConvBlock(num_features, num_features * 4, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.PixelShuffle(2),
            ConvBlock(num_features, out_channels, kSize, padding=(kSize - 1) // 2, stride=1)
        ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        out = self.UPNet(x)

        return out

class AsyCA(nn.Module):
    def __init__(self, num_features, ratio):
        super(AsyCA, self).__init__()
        self.out_channels = num_features
        self.conv_init = ConvBlock(num_features * 2, num_features, kernel_size=1, padding=0, stride=1)
        self.conv_dc = ConvBlock(num_features, num_features // ratio, kernel_size=1, padding=0, stride=1)
        self.conv_ic = ConvBlock(num_features // ratio, num_features * 2, kernel_size=1, padding=0, stride=1)
        self.act = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        batch_size = x1.size(0)

        feat_init = torch.cat((x1, x2), 1)
        feat_init = self.conv_init(feat_init)
        fea_avg = self.avg_pool(feat_init)
        feat_ca = self.conv_dc(fea_avg)
        feat_ca = self.conv_ic(self.act(feat_ca))

        a_b = feat_ca.reshape(batch_size, 2, self.out_channels, -1)

        a_b = self.softmax(a_b)
        # print(a_b[0,0,0,0],)
        a_b = list(a_b.chunk(2, dim=1))  # split to a and b
        a_b = list(map(lambda x1: x1.reshape(batch_size, self.out_channels, 1, 1), a_b))
        self.V1 =V1= a_b[0] * x1
        self.V2 =V2= a_b[1] * x2
        V = V1 + V2
        return V
        
class TOPAL(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, growthrate=32):
        super(TOPAL, self).__init__()

        self.model1 = MSBDN()

        self.model2 = deepWBnet()


        kSize = 3
        ratio = 4

        self.feat_conv1 = ConvBlock(in_channels, num_features, kSize, padding=(kSize - 1) // 2, stride=1)
        self.feat_conv2 = ConvBlock(in_channels, num_features, kSize, padding=(kSize - 1) // 2, stride=1)
        self.RDB0 = RDB(num_features, growthrate, kSize)
        self.RDB1 = RDB(num_features, growthrate, kSize)
        self.RDB2 = RDB(num_features, growthrate, kSize)

        self.AsyCA1 = AsyCA(num_features, ratio)

        self.out_conv = ConvBlock(num_features, out_channels, 1, padding=0, stride=1)

    def forward(self, x):

        pre1 = self.model1(x)
        pre2 = self.model2(x)

        x1 = self.feat_conv1(pre1)
        self.x1 = x1 = self.RDB0(x1)

        x2 = self.feat_conv2(pre2)
        self.x2 = x2 = self.RDB1(x2)

        self.fused = x = self.AsyCA1(x1, x2)

        x = self.RDB2(x)
        x = self.out_conv(x)
        return x.clamp(0, 1)

if __name__ == '__main__':
    model = TOPAL(in_channels= 3, out_channels = 3, num_features = 64, growthrate = 32).cuda()
    t = torch.randn(1, 3, 256, 256).cuda()
    res = model(t)
    print(res.shape)