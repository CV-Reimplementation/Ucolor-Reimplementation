import torch
import torch.nn as nn

class WaterNet(nn.Module):
    def __init__(self):
        super(WaterNet, self).__init__()
        self._init_layers()

    def _init_layers(self):
        self.conv2wb_1 = nn.Conv2d(12, 128, 7, 1, 3)
        self.conv2wb_1_relu = nn.ReLU(inplace=True)

        self.conv2wb_2 = nn.Conv2d(128, 128, 5, 1, 2)
        self.conv2wb_2_relu = nn.ReLU(inplace=True)

        self.conv2wb_3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2wb_3_relu = nn.ReLU(inplace=True)

        self.conv2wb_4 = nn.Conv2d(128, 64, 1, 1, 0)
        self.conv2wb_4_relu = nn.ReLU(inplace=True)

        self.conv2wb_5 = nn.Conv2d(64, 64, 7, 1, 3)
        self.conv2wb_5_relu = nn.ReLU(inplace=True)

        self.conv2wb_6 = nn.Conv2d(64, 64, 5, 1, 2)
        self.conv2wb_6_relu = nn.ReLU(inplace=True)

        self.conv2wb_7 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2wb_7_relu = nn.ReLU(inplace=True)

        self.conv2wb_77 = nn.Conv2d(64, 3, 3, 1, 1)
        self.conv2wb_77_sigmoid = nn.Sigmoid()

        # wb
        self.conv2wb_9 = nn.Conv2d(6, 32, 7, 1, 3)
        self.conv2wb_9_relu = nn.ReLU(inplace=True)

        self.conv2wb_10 = nn.Conv2d(32, 32, 5, 1, 2)
        self.conv2wb_10_relu = nn.ReLU(inplace=True)

        self.conv2wb_11 = nn.Conv2d(32, 3, 3, 1, 1)
        self.conv2wb_11_relu = nn.ReLU(inplace=True)

        # ce
        self.conv2wb_99 = nn.Conv2d(6, 32, 7, 1, 3)
        self.conv2wb_99_relu = nn.ReLU(inplace=True)

        self.conv2wb_100 = nn.Conv2d(32, 32, 5, 1, 2)
        self.conv2wb_100_relu = nn.ReLU(inplace=True)

        self.conv2wb_111 = nn.Conv2d(32, 3, 3, 1, 1)
        self.conv2wb_111_relu = nn.ReLU(inplace=True)

        # gc
        self.conv2wb_999 = nn.Conv2d(6, 32, 7, 1, 3)
        self.conv2wb_999_relu = nn.ReLU(inplace=True)

        self.conv2wb_1000 = nn.Conv2d(32, 32, 5, 1, 2)
        self.conv2wb_1000_relu = nn.ReLU(inplace=True)

        self.conv2wb_1111 = nn.Conv2d(32, 3, 3, 1, 1)
        self.conv2wb_1111_relu = nn.ReLU(inplace=True)


    def forward(self, x, wb, ce, gc):
        conb0 = torch.cat([x, wb, ce, gc], dim=1)
        conv_wb1 = self.conv2wb_1_relu(self.conv2wb_1(conb0))
        conv_wb2 = self.conv2wb_2_relu(self.conv2wb_2(conv_wb1))
        conv_wb3 = self.conv2wb_3_relu(self.conv2wb_3(conv_wb2))
        conv_wb4 = self.conv2wb_4_relu(self.conv2wb_4(conv_wb3))
        conv_wb5 = self.conv2wb_5_relu(self.conv2wb_5(conv_wb4))
        conv_wb6 = self.conv2wb_6_relu(self.conv2wb_6(conv_wb5))
        conv_wb7 = self.conv2wb_7_relu(self.conv2wb_7(conv_wb6))
        conv_wb77 = self.conv2wb_77_sigmoid(self.conv2wb_77(conv_wb7))

        # wb
        conb00 = torch.cat([x, wb], dim=1)
        conv_wb9 = self.conv2wb_9_relu(self.conv2wb_9(conb00))
        conv_wb10 = self.conv2wb_10_relu(self.conv2wb_10(conv_wb9))
        wb1 = self.conv2wb_11_relu(self.conv2wb_11(conv_wb10))

        # ce
        conb11 = torch.cat([x, ce], dim=1)
        conv_wb99 = self.conv2wb_99_relu(self.conv2wb_99(conb11))
        conv_wb100 = self.conv2wb_100_relu(self.conv2wb_100(conv_wb99))
        ce1 = self.conv2wb_111_relu(self.conv2wb_111(conv_wb100))

        # gc
        conb111 = torch.cat([x, gc], dim=1)
        conv_wb999 = self.conv2wb_999_relu(self.conv2wb_999(conb111))
        conv_wb1000 = self.conv2wb_1000_relu(self.conv2wb_1000(conv_wb999))
        gc1 = self.conv2wb_1111_relu(self.conv2wb_1111(conv_wb1000))

        weight_wb, weight_ce, weight_gc = conv_wb77[:, 0:1, :, :], conv_wb77[:, 1:2, :, :], conv_wb77[:, 2:3, :, :]
        out = (weight_wb * wb1) + (weight_ce * ce1) + (weight_gc * gc1)

        return out


if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256).cuda()
    model = WaterNet().cuda()
    res = model(t, t, t, t)
    print(res.shape)
    