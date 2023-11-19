import torch
from torch import nn


# 论文：10.1109/TCSVT.2023.3237993
# https://ieeexplore.ieee.org/abstract/document/10019314
# UIALN: Enhancement for Underwater Image with Artificial Light
class Retinex_Decomposition_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(Retinex_Decomposition_net, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        # relu激活
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        return x


class Illumination_Correction(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(Illumination_Correction, self).__init__()
        self.down_1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)
        self.down_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.down_3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.up_1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.up_2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.up_3 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, output_padding=1)
        # 相当于两次反卷积
        self.up_4_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)  ################# 存疑
        self.up_4_2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1)
        self.up_5 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, output_padding=1)
        self.conv1 = nn.Conv2d(32 * 3, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.down_3(x)
        x1 = self.up_1(x)
        print(x1.shape)
        x2 = self.up_2(x1)
        print(x2.shape)
        x = self.up_3(x2)
        x1 = self.up_4_2(self.up_4_1(x1))
        x2 = self.up_5(x2)
        print(x.shape, x1.shape, x2.shape)
        x = torch.cat((x, x1, x2), dim=1)
        x = self.conv1(x)
        return x


# Residual Dense Block
class Dense_Block_IN(nn.Module):
    def __init__(self, block_num, inter_channel, channel, with_residual=True):
        super(Dense_Block_IN, self).__init__()
        concat_channels = channel + block_num * inter_channel
        channels_now = channel

        self.group_list = nn.ModuleList([])
        for i in range(block_num):
            group = nn.Sequential(
                nn.Conv2d(in_channels=channels_now, out_channels=inter_channel, kernel_size=3,
                          stride=1, padding=1),
                nn.InstanceNorm2d(inter_channel, affine=True),
                nn.ReLU(),
            )
            self.add_module(name='group_%d' % i, module=group)
            self.group_list.append(group)
            channels_now += inter_channel
        assert channels_now == concat_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_channels, channel, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(channel, affine=True),
            nn.ReLU(),
        )
        self.with_residual = with_residual

    def forward(self, x):
        feature_list = [x]
        for group in self.group_list:
            inputs = torch.cat(feature_list, dim=1)
            outputs = group(inputs)
            feature_list.append(outputs)
        inputs = torch.cat(feature_list, dim=1)
        fusion_outputs = self.fusion(inputs)
        if self.with_residual:
            block_outputs = fusion_outputs + x
        else:
            block_outputs = fusion_outputs

        return block_outputs


class AL_Area_Selfguidance_Color_Correction(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(AL_Area_Selfguidance_Color_Correction, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.RDB1 = Dense_Block_IN(4, 32, 64)
        self.Down_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.RDB2 = Dense_Block_IN(4, 32, 128)
        self.Down_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.RDB3 = Dense_Block_IN(4, 32, 256)
        self.Up_1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.RDB4 = Dense_Block_IN(4, 32, 128 + 128)
        self.Up_2 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=3, stride=2, output_padding=1)
        self.RDB5 = Dense_Block_IN(4, 32, 64 + 64)
        self.conv3 = nn.Conv2d(64 + 64, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        x = x * y
        x = self.conv1(x)
        y = self.conv2(y)
        x = torch.cat((x, y), dim=1)
        x1 = self.RDB1(x)
        x2 = self.RDB2(self.Down_1(x1))
        x = self.RDB3(self.Down_2(x2))
        x = self.Up_1(x)
        x = torch.cat((x, x2), dim=1)
        x = self.Up_2(self.RDB4(x))
        x = torch.cat((x, x1), dim=1)
        x = self.RDB5(x)
        x = self.conv3(x)
        return x


class Detail_Enhancement(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Detail_Enhancement, self).__init__()
        self.Down_1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)
        self.DB_1 = Dense_Block_IN(4, 32, 32, with_residual=False)
        self.Down_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.DB_2 = Dense_Block_IN(4, 32, 64, with_residual=False)
        self.Down_3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.DB_3 = Dense_Block_IN(4, 32, 128, with_residual=False)
        self.Down_4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.DB_4 = Dense_Block_IN(4, 32, 256, with_residual=False)
        self.Up_1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.DB_5 = Dense_Block_IN(4, 32, 128 + 128, with_residual=False)
        self.Up_2 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=3, stride=2)
        self.DB_6 = Dense_Block_IN(4, 32, 64 + 64, with_residual=False)
        self.Up_3 = nn.ConvTranspose2d(64 + 64, 32, kernel_size=3, stride=2)
        self.DB_7 = Dense_Block_IN(4, 32, 32 + 32, with_residual=False)
        self.Up_4 = nn.ConvTranspose2d(32 + 32, 16, kernel_size=3, stride=2, output_padding=1)
        self.DB_8 = Dense_Block_IN(4, 32, 16 + in_channels, with_residual=False)
        self.conv1 = nn.Conv2d(16 + in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x0 = x
        x1 = self.DB_1(self.Down_1(x))
        x2 = self.DB_2(self.Down_2(x1))
        x3 = self.DB_3(self.Down_3(x2))
        x = self.DB_4(self.Down_4(x3))
        x = self.Up_1(x)
        x = torch.cat((x, x3), dim=1)
        x = self.Up_2(self.DB_5(x))
        x = torch.cat((x, x2), dim=1)
        x = self.Up_3(self.DB_6(x))
        x = torch.cat((x, x1), dim=1)
        x = self.Up_4(self.DB_7(x))
        x = torch.cat((x, x0), dim=1)
        x = self.DB_8(x)
        x = self.conv1(x)
        return x


class Channels_Fusion(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Channels_Fusion, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Detail_Enhancement().to(device)
    t = torch.randn(1, 3, 256, 256).to(device)
    res = model(t)
    print(res.shape)