import torch.nn as nn
import torch
from torch.nn import init


class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):
    def __init__(self, channel=512,reduction=16,kernel_size=7):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual

class ResnetBlock(nn.Module):
    """Resnet block.

    Adapted from "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix".
    """

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block.

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class RauneNet(nn.Module):
    """Residual and Attention-driven underwater enhancement Network.
    """
    def __init__(self, input_nc=3, output_nc=3, n_blocks=30, n_down=2, ngf=64,
                 padding_type='reflect', use_dropout=False, use_att_down=True, use_att_up=False,
                 norm_layer=nn.InstanceNorm2d):
        """Initializes the RAUNE-Net.

        Args:
            input_nc: Number of channels of input images.
            output_nc: Number of chnnels of output images.
            n_blocks: Number of residual blocks.
            n_down: Number of down-sampling blocks.
            ngf: Number of kernels of Conv2d layer in `WRPM`.
            padding_type: Type of padding layer in Residual Block.
            use_dropout: Whether to use dropout.
            use_att_down: Whether to use attention block in down-sampling.
            use_att_up: Whether to use attention block in up-sampling.
            norm_layer: Type of Normalization layer.
        """
        assert (n_blocks >= 0 and n_down >= 0)
        super().__init__()
        use_bias = False if norm_layer else True

        model = []

        # Wide-range Perception Module (WRPM)
        model.append(nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)
        ))

        # Attention Down-sampling Module (ADM)
        for i in range(n_down):
            mult = 2 ** i
            model.append(self._down(ngf*mult, ngf*mult*2, norm_layer=norm_layer, use_att=use_att_down, use_dropout=use_dropout))
        
        # High-level Features Residual Learning Module (HFRLM)
        mult = 2 ** n_down
        for i in range(n_blocks):
            model.append(ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                     use_dropout=use_dropout, use_bias=use_bias))
        
        # Up-sampling Module (UM)
        for i in range(n_down):
            mult = 2 ** (n_down - i)
            model.append(self._up(ngf * mult, int(ngf * mult / 2), use_att=use_att_up, use_dropout=use_dropout))

        # Feature Map Smoothing Module (FMSM) and Tanh Activation Layer
        model.append(nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ))
        
        self.model = nn.Sequential(*model)

    def _down(self, in_channels, out_channels, norm_layer=None, use_att=True, use_dropout=False, dropout_rate=0.5):
        """Attention Down-sampling Block.

        Args:
            in_channels: Number of channels of input tensor.
            out_channels: Number of channels of output tensor.
            norm_layer: Type of Normalization layer.
            use_att: Whether to use attention.
            use_dropout: Whether to use dropout.
            dropout_rate: Probability of dropout layer.
        """
        use_bias = False if norm_layer else True
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=use_bias)]
        if norm_layer:
            layers.append(norm_layer(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        if use_att:
            layers.append(CBAMBlock(out_channels))
        return nn.Sequential(*layers)

    def _up(self, in_channels, out_channels, use_att=False, use_dropout=False, dropout_rate=0.5):
        """Up-sampling Block.

        Args:
            in_channels: Number of channels of input tensor.
            out_channels: Number of channels of output tensor.
            use_att: Whether to use attention.
            use_dropout: Whether to use dropout.
            dropout_rate: Probability of dropout layer.
        """
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
        ]
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        if use_att:
            layers.append(CBAMBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        """Forward function.

        Args:
            input: Input images. Type of `torch.Tensor`.
        """
        return self.model(input)
    

if __name__ == '__main__':
    inp = torch.randn(1, 3, 256, 256).cuda()
    model = RauneNet().cuda()
    res = model(inp)
    print(res.shape)