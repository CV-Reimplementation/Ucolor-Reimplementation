import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import os

class Attention(nn.Module):
    def __init__(self,in_planes,ratio,K,temprature=30,init_weight=True):
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.temprature=temprature
        assert in_planes>ratio
        hidden_planes=in_planes//ratio
        self.net=nn.Sequential(
            nn.Conv2d(in_planes,hidden_planes,kernel_size=1,bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes,K,kernel_size=1,bias=False)
        )

        if(init_weight):
            self._initialize_weights()

    def update_temprature(self):
        if(self.temprature>1):
            self.temprature-=1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        att=self.avgpool(x) #bs,dim,1,1
        att=self.net(att).view(x.shape[0],-1) #bs,K
        return F.softmax(att/self.temprature,-1)

class DynamicConv(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride,padding=0,dilation=1,grounps=1,bias=True,K=4,temprature=30,ratio=2,init_weight=True):
        super().__init__()
        self.in_planes=in_planes
        self.out_planes=out_planes
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.groups=grounps
        self.bias=bias
        self.K=K
        self.init_weight=init_weight
        self.attention=Attention(in_planes=in_planes,ratio=ratio,K=K,temprature=temprature,init_weight=init_weight)

        self.weight=nn.Parameter(torch.randn(K,out_planes,in_planes//grounps,kernel_size,kernel_size),requires_grad=True)
        if(bias):
            self.bias=nn.Parameter(torch.randn(K,out_planes),requires_grad=True)
        else:
            self.bias=None
        
        if(self.init_weight):
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self,x):
        bs,in_planels,h,w=x.shape
        softmax_att=self.attention(x) #bs,K
        x=x.view(1,-1,h,w)
        weight=self.weight.view(self.K,-1) #K,-1
        aggregate_weight=torch.mm(softmax_att,weight).view(bs*self.out_planes,self.in_planes//self.groups,self.kernel_size,self.kernel_size) #bs*out_p,in_p,k,k

        if(self.bias is not None):
            bias=self.bias.view(self.K,-1) #K,out_p
            aggregate_bias=torch.mm(softmax_att,bias).view(-1) #bs,out_p
            output=F.conv2d(x,weight=aggregate_weight,bias=aggregate_bias,stride=self.stride,padding=self.padding,groups=self.groups*bs,dilation=self.dilation)
        else:
            output=F.conv2d(x,weight=aggregate_weight,bias=None,stride=self.stride,padding=self.padding,groups=self.groups*bs,dilation=self.dilation)
        
        output=output.view(bs,self.out_planes,h,w)
        return output
    
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        input = input.view(input.shape[0], -1)
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            # out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feat, kernel_size=3, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(BaseConv2d(in_channel=n_feat, out_channel=4 * n_feat, kernel_size=kernel_size, stride=1,
                                    padding=(kernel_size // 2), bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(BaseConv2d(in_channel=n_feat, out_channel=9 * n_feat, kernel_size=kernel_size, stride=1,
                                padding=(kernel_size // 2), bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class BaseConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input, condition_feature=None):
        '''
        input_features [batch, 64, h, w]
        condition_features [batch, 1, in_channel=64, 1, 1]
        '''
        b, c, h, w = input.shape
        if c != self.in_channel:
            raise ValueError('Input channel is not equal with conv in_channel')
        if condition_feature != None:
            # [batch, out_channel, in_channel, self.kernel_size, self.kernel_size] = [batch, 64, 64, 3, 3]
            weight = self.weight.unsqueeze(0) * self.scale * condition_feature
            weight = weight.view(b * self.in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            input = input.view(1, b * self.in_channel, h, w)
            bias = torch.repeat_interleave(self.bias, repeats=b, dim=0)
            out = F.conv2d(input, weight, bias=bias, stride=self.stride, padding=self.padding, groups=b)
            _, _, height, width = out.shape
            out = out.view(b, self.out_channel, height, width)
        else:
            out = F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class MFE(nn.Module):
    def __init__(self, in_channels=16, out_channels=16):
        super(MFE, self).__init__()

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

        self.conv3_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.conv5_5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        )

        # 全局最大池化
        self.GlobalMaximumPolling = nn.AdaptiveMaxPool2d(1)
        self.ChannelAttention=ChannelAttention(64)

        # 全连接层
        self.FC1 = nn.Sequential(
            nn.Linear(1, 1, bias=True),
            nn.Linear(1, 1, bias=True)
        )

    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.conv3_3(x)
        x3 = self.conv5_5(x)

        # xll = x1 + x2 + x3
        xll = self.ChannelAttention(x)

        C1 = self.FC1(xll)
        C2 = self.FC1(xll)
        C3 = self.FC1(xll)

        C1X1 = C1 * x1
        C2X2 = C2 * x2
        C3X3 = C3 * x3

        Y = C1X1 + C2X2 + C3X3

        return Y


class ResBlock(nn.Module):
    '''Residual block with controllable residual connections
condition feature----------------
                   |            |
              modulation    modulation
                   |            |
              ---Conv—--ReLU--Conv-+--
                 |_________________|
    '''

    def __init__(self, nf=64, kernel_size=3):
        super(ResBlock, self).__init__()
        self.conv1 = BaseConv2d(in_channel=nf, out_channel=nf, kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2, bias=True)
        self.conv2 = BaseConv2d(in_channel=nf, out_channel=nf, kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2, bias=True)
        self.act = nn.ReLU(inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, input_features, condition_features=None):
        '''
        input_features [batch, 64, h, w]
        condition_features [n_conv_each_block, batch, 1, 64, 1, 1]
        '''

        res = self.conv1(input_features, condition_features[0])
        res = self.act(res)
        res = self.conv2(res, condition_features[1])

        out = res + input_features
        return out




class PMS(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(PMS, self).__init__()

        self.res1 = ResBlock(nf=64, kernel_size=3)
        self.res2 = ResBlock(nf=64, kernel_size=3)
        self.res3 = ResBlock(nf=64, kernel_size=3)
        self.res4 = ResBlock(nf=64, kernel_size=3)
        self.res5 = ResBlock(nf=64, kernel_size=3)
        self.res6 = ResBlock(nf=64, kernel_size=3)
        self.res7 = ResBlock(nf=64, kernel_size=3)
        self.res8 = ResBlock(nf=64, kernel_size=3)
        self.res9 = ResBlock(nf=64, kernel_size=3)

        self.mfe1 = MFE(64, 64)
        self.mfe2 = MFE(64, 64)
        self.mfe3 = MFE(64, 64)
        self.mfe4 = MFE(64, 64)
        self.mfe5 = MFE(64, 64)


        # self.Tdec = Tdec()
        # self.pupu = nn.Conv2d(512, 1024, 3, 1, 1)
        # self.U1 = UpSampling(16)

    # 前向计算，输出一张与原图相同尺寸的图片矩阵
    def forward(self, x, condition_feature1,condition_feature2,map):
        e1 = self.res1(x, condition_feature1[0])
        e1 = self.mfe1(e1)
        e1 = self.res1(e1, condition_feature1[0])

        e2 = self.res2(e1, condition_feature1[1])
        e2 = self.res2(e2, condition_feature1[1])
        e2 = self.res2(e2, condition_feature1[1])

        e3 = self.res3(e2, condition_feature1[2])
        e3 = self.mfe2(e3)
        e3 = self.res3(e3, condition_feature1[2])

        e4 = self.res4(e3, condition_feature1[3])
        e4 = self.res4(e4, condition_feature1[3])
        e4 = self.res4(e4, condition_feature1[3])

        # e4=e4*map+e4
        e5 = self.res5(e4, condition_feature2[0])
        e5 = self.mfe3(e5)
        e5 = e5 * map + e5  #[SSIM 0.861134] , [PSNR: 21.035865] , [MSE: 712.630818]
        e5 = self.res5(e5, condition_feature2[0])


        e5 = e5 + e4
        d6 = self.res6(e5, condition_feature1[5])
        d6 = self.res6(d6, condition_feature1[5])
        d6 = self.res6(d6, condition_feature1[5])

        d7 = e3 + d6
        d7 = self.res7(d7, condition_feature1[6])
        d7 = self.mfe4(d7)
        d7 = self.res7(d7, condition_feature1[6])

        d8 = e2 + d7
        d8 = self.res8(d8, condition_feature1[7])
        d8 = self.res8(d8, condition_feature1[7])
        d8 = self.res8(d8, condition_feature1[7])

        d9 = e1 + d8
        d9 = self.res9(d9, condition_feature1[8])
        d9 = self.mfe5(d9)
        d9 = self.res9(d9, condition_feature1[8])

        return d9


class PMS_1(nn.Module):
    def __init__(self, scale=4, input_channels=3, channels=64, n_block=10, n_conv_each_block=2, kernel_size=3, conv_index='22'):
        super(PMS_1, self).__init__()
        # self.args = args
        self.scale = scale
        self.input_channels = input_channels
        self.channels = channels
        self.n_block = n_block
        self.n_conv_each_block = n_conv_each_block
        self.conv_index = conv_index

        self.input_conv = BaseConv2d(in_channel=input_channels, out_channel=channels, kernel_size=kernel_size, stride=1,
                                     padding=kernel_size // 2, bias=True)
        layers = []
        for _ in range(n_block):
            layers.append(ResBlock(nf=channels, kernel_size=kernel_size))
        self.backbone = nn.Sequential(*layers)
        self.upsampler = Upsampler(scale, channels, act=False)
        self.output_conv = BaseConv2d(in_channel=channels, out_channel=input_channels, kernel_size=kernel_size,
                                      stride=1, padding=kernel_size // 2, bias=True)

        self.modulations1 = Modulations(n_block=self.n_block, n_conv_each_block=self.n_conv_each_block,
                                       conv_index="128",
                                       sr_in_channel=self.channels)
        self.modulations2 = Modulations(n_block=self.n_block, n_conv_each_block=self.n_conv_each_block,
                                       conv_index="128",
                                       sr_in_channel=self.channels)
        self.PMS = PMS()

        self.conv = nn.Conv2d(48, 64, 3, 1, 1)


    def forward(self, x, f_ads,f_tds,map):
        '''
        x [batch_size*support_size, 3, h, w]
        condition_features [batch_size, 128, 1, 1]

        modulated_condition_features [n_block, n_conv_each_block, batch_size, 64, 1, 1, 1]
        '''

        b, _, h, w = x.shape
        f_ads = self.modulations1(
            f_ads)  # [n_block, n_conv_each_block, batch_size, 1, `128`, 1, 1]
        f_ads = torch.repeat_interleave(f_ads, repeats=b // f_ads.shape[2],
                                                    dim=2)  # [n_block, n_conv_each_block, batch, 1, 128, 1, 1]

        f_tds = self.modulations2(
            f_tds)  # [n_block, n_conv_each_block, batch_size, 1, 64, 1, 1]
        f_tds = torch.repeat_interleave(f_tds, repeats=b // f_tds.shape[2],
                                                    dim=2)  # [n_block, n_conv_each_block, batch, 1, 64, 1, 1]


        map = 1 - map  # (1,1,256,256)
        map24 = map.repeat(1, 8, 1, 1)
        map48 = torch.cat([map24, map24], dim=1)  # 1,48,256,256
        map64 = self.conv(map48)

        x = self.input_conv(x)
        x = self.PMS(x, f_ads,f_tds,map64)
        x = self.output_conv(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))


class ConditionNet(nn.Module):
    def __init__(self, n_block=10, n_conv_each_block=2, conv_index='22', sr_in_channel=64, support_size=10):
        super(ConditionNet, self).__init__()
        self.support_size = support_size
        self.n_block = n_block
        self.n_conv_each_block = n_conv_each_block
        self.n_modulation = n_block * n_conv_each_block
        self.conv_index = conv_index
        if conv_index == '128':
            self.condition_channel = 128
        elif self.conv_index == '64':
            self.condition_channel = 64
        else:
            raise ValueError('Illegal VGG conv_index!!!')
        self.sr_in_channel = sr_in_channel


        self.condition = self.get_VGG_condition()
        initialize_weights(self.condition, 0.1)

        # self.dqg = dqg()

    def get_VGG_condition(self):

        cfg = [64, 64, 'P', 128, 128, 'P']
        if self.conv_index == '128':
            cfg_idx = cfg[:5]
        elif self.conv_index == '64':
            cfg_idx = cfg[:2]
        else:
            raise ValueError('Illegal VGG conv_index!!!')
        return self._make_layers(cfg_idx)

    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3 * self.support_size
        for v in cfg:
            if v == 'P':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d =DynamicConv(in_planes=in_channels, out_planes=v, kernel_size=3, stride=1, padding=1, bias=False)
                # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def reset_support_size(self, support_size):
        self.support_size = support_size

    def forward(self, support_x):
        '''return batch_size condition_features
        Input:
        For training
        support_x [batch_size, support_size*3, h, w]
        For testing batch_size = 1
        support_x [1, support_size*3, h, w]

        '''
        # todo daqiguang
        # support_x = self.dqg(support_x)

        support_conditional_feature = self.condition(support_x)  # [batch_size, 128, h/2, w/2]

        _, _, h, w = support_conditional_feature.shape
        conditional_feature = F.avg_pool2d(support_conditional_feature, kernel_size=h,
                                           stride=w)  # [batch_size, 128, 1, 1]
        return conditional_feature


class Modulations(nn.Module):
    def __init__(self, n_block=10, n_conv_each_block=2, conv_index='128', sr_in_channel=64):
        super(Modulations, self).__init__()
        self.n_block = n_block
        self.n_conv_each_block = n_conv_each_block
        self.n_modulation = n_block * n_conv_each_block
        self.conv_index = conv_index
        if conv_index == '128':
            self.condition_channel = 128
        elif self.conv_index == '64':
            self.condition_channel = 256
        else:
            raise ValueError('Illegal VGG conv_index!!!')
        self.sr_in_channel = sr_in_channel

        self.modulations = self.get_linear_modulations()
        initialize_weights(self.modulations, 0.1)

    def get_linear_modulations(self):
        modules = []
        for _ in range(self.n_modulation):
            modules.append(EqualLinear(self.condition_channel, self.sr_in_channel, bias_init=1))

        return nn.Sequential(*modules)

    def forward(self, condition_feature):
        '''
        Input:
        For training
        condition_feature:[batch_size, 128, 1, 1]
        For testing
        condition_feature:[1, 128, 1, 1]

        repeat n_block*2 condition_features [n_block*n_conv_each_block, batch_size, 128, 1, 3, 3]
        for i in range n_block*2:
            EqualLinear modulation condition_features[i] [batch_size, 1, 64, 1, 1]
        condition_features [n_block, n_conv_each_block, batch_size, 1, 64, 1, 1]
        '''
        batch_size, condition_channel, h, w = condition_feature.shape
        if condition_channel != self.condition_channel:
            raise ValueError('the shape of input condition_feature should be [batch_size, condition_channel, h, w]')

        condition_weight = []
        repeat_support_feature = torch.repeat_interleave(condition_feature.unsqueeze(0), repeats=self.n_modulation,
                                                         dim=0)  # [n_block*2, batch_size, 128, 1, 1]
        for idx, modulation in enumerate(self.modulations):
            cur_support_feature = repeat_support_feature[idx]
            reshape_condition_feature = modulation(cur_support_feature.permute(0, 2, 3, 1)).view(batch_size, 1,
                                                                                                 self.sr_in_channel, 1,
                                                                                                 1)
            condition_weight.append(reshape_condition_feature.unsqueeze(0))

        out_features = torch.cat(condition_weight, 0).to(condition_feature.device)
        out_features = out_features.view(self.n_block, self.n_conv_each_block, batch_size, 1, self.sr_in_channel, 1, 1)

        return out_features


class GUPDM(nn.Module):
    def __init__(self):
        super(GUPDM, self).__init__()

        self.scale = 1
        self.support_size = 1

        self.input_channels = 3
        self.kernel_size = 3
        self.channels = 64
        self.n_block1 = 9
        self.n_block2 = 4
        self.n_conv_each_block = 2
        self.conv_index = 22

        self.PMS = PMS_1(scale=self.scale,
                                  input_channels=self.input_channels,
                                  channels=self.channels, n_block=self.n_block1,
                                  n_conv_each_block=self.n_conv_each_block, kernel_size=self.kernel_size)

        self.ADS = ConditionNet(n_block=self.n_block1, n_conv_each_block=self.n_conv_each_block,
                                          conv_index="128",
                                          sr_in_channel=self.channels, support_size=self.support_size)

        self.TDS = ConditionNet(n_block=self.n_block2, n_conv_each_block=self.n_conv_each_block,
                                          conv_index="128",
                                          sr_in_channel=self.channels, support_size=self.support_size)

    def _load_sr_net(self):
        if os.path.exists(self.args.pretrained_model_checkpoint_dir + self.args.pretrained_sr_net_path.split("/")[-1]):
            print('loading pretrained model : {}'.format(
                self.args.pretrained_model_checkpoint_dir + self.args.pretrained_sr_net_path.split("/")[-1]))
        else:
            raise ValueError('Please get the pretrained BaseNet')
        self.sr_net.load_state_dict(
            torch.load(self.args.pretrained_model_checkpoint_dir + self.args.pretrained_sr_net_path.split("/")[-1],
                       map_location='cpu'), strict=True)

    def _load_pretrain_net(self):
        if os.path.exists(self.args.pretrained_model_checkpoint_dir + self.args.load_trained_model_path.split("/")[-1]):
            print('loading pretrained model : {}'.format(
                self.args.pretrained_model_checkpoint_dir + self.args.load_trained_model_path.split("/")[-1]))
        else:
            raise ValueError('Please get the pretrained CMDSR')
        self.load_state_dict(
            torch.load(self.args.pretrained_model_checkpoint_dir + self.args.load_trained_model_path.split("/")[-1],
                       map_location='cpu'), strict=True)

    def reset_support_size(self, support_size=6):
        self.support_size = support_size
        self.condition_net.reset_support_size(support_size)

    def forward(self, x, map):
        f_ads = self.ADS(x)
        f_tds = self.TDS(x)
        x = self.PMS(x, f_ads, f_tds, map)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))



if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256).cuda()

    net = GUPDM().cuda()
    y = net(x, x)
    print(y.shape)