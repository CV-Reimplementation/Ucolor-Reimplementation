import torch
import torch.nn as nn
from kornia.color import rgb_to_lab, rgb_to_hsv


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lrelu = nn.LeakyReLU(inplace=True)

        # Depth conv
        self.depth1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.depth2 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.depth3 = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=3, stride=1, padding=1)
        # self.MaxPool2d = nn.AdaptiveMaxPool2d(1)
        # first encoder
        # 256*256
        # HSV
        self.conv_pan1_0 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_pan1_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_pan1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_pan1_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv_pan1_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_pan1_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_pan1_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_pan1_7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        #rgb
        self.conv_ms1_0 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_ms1_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_ms1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_ms1_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv_ms1_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_ms1_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_ms1_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_ms1_7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        #lab
        self.conv_hs1_0 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_hs1_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_hs1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_hs1_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv_hs1_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_hs1_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_hs1_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_hs1_7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # second encoder
        # 128*128 hsv
        self.conv_pan2_0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_pan2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_pan2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_pan2_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_pan2_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_pan2_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_pan2_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_pan2_7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 128*128 rgb
        self.conv_ms2_0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_ms2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_ms2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_ms2_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_ms2_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_ms2_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_ms2_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_ms2_7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 128*128 lab
        self.conv_hs2_0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_hs2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_hs2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_hs2_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_hs2_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_hs2_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_hs2_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_hs2_7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # third encoder
        # hsv
        self.conv_pan3_0 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_pan3_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_pan3_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_pan3_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_pan3_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_pan3_5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_pan3_6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_pan3_7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        #rgb
        self.conv_ms3_0 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_ms3_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_ms3_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_ms3_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_ms3_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_ms3_5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_ms3_6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_ms3_7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        #lab
        self.conv_hs3_0 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_hs3_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_hs3_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_hs3_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_hs3_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_hs3_5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_hs3_6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_hs3_7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        # cat conv
        self.conv_cat_1 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_cat_2 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_cat_3 = nn.Conv2d(in_channels=1536, out_channels=512, kernel_size=3, stride=1, padding=1)

        #######decoder
        # first
        self.conv_de1_0 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_de1_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_de1_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_de1_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_de1_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_de1_5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_de1_6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_de1_7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # self.conv_de1_8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        # second
        self.conv_de2_0 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_de2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_de2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_de2_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_de2_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_de2_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_de2_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_de2_7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # self.conv_de2_8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # third
        self.conv_de3_0 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_de3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_de3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_de3_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_de3_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_de3_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_de3_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv_de3_7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # self.conv_de3_8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv_de4 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, padding=0)

        self.se_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(384, 384 // 16, kernel_size=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(384 // 16, 384, kernel_size=1, padding=0)
        )
        self.se_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(768, 768 // 16, kernel_size=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(768 // 16, 768, kernel_size=1, padding=0)
        )
        self.se_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1536, 1536 // 16, kernel_size=1, padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1536 // 16, 1536, kernel_size=1, padding=0)
        )
    
    def forward(self, rgb, _depth):
        
        hsv = rgb_to_hsv(rgb)
        lab = rgb_to_lab(rgb)

        depth_2down = torch.max_pool2d(1 - _depth,kernel_size=2,stride=2,padding=0)
        depth_4down = torch.max_pool2d(depth_2down,kernel_size=2,stride=2,padding=0)

        _depth = self.lrelu(self.depth1(_depth))
        depth_2down = self.lrelu(self.depth2(depth_2down))
        depth_4down = self.lrelu(self.depth3(depth_4down))

        ##########encoder#########
        # first encoder
        pan1_0 = self.lrelu(self.conv_pan1_0(hsv))  # 3==>128
        pan1_1 = self.lrelu(self.conv_pan1_1(pan1_0))  # 128>128
        pan1_2 = self.lrelu(self.conv_pan1_2(pan1_1))  # 128>128
        pan1_3 = self.conv_pan1_3(pan1_2)
        pan1_add_1 = torch.add(pan1_0, pan1_3)  # 128>128  第一个残差块
        pan1_4 = self.lrelu(self.conv_pan1_4(pan1_add_1))  # 128>128
        pan1_5 = self.lrelu(self.conv_pan1_5(pan1_4))  # 128>128
        pan1_6 = self.lrelu(self.conv_pan1_6(pan1_5))
        pan1_7 = self.conv_pan1_7(pan1_6)
        F_hsv11 = torch.add(pan1_4, pan1_7)
        F_hsv1 = torch.max_pool2d(F_hsv11, kernel_size=2, stride=2, padding=0)
        # second encoder
        pan2_0 = self.lrelu(self.conv_pan2_0(F_hsv1))
        pan2_1 = self.lrelu(self.conv_pan2_1(pan2_0))
        pan2_2 = self.lrelu(self.conv_pan2_2(pan2_1))
        pan2_3 = self.conv_pan2_3(pan2_2)
        pan2_add_1 = torch.add(pan2_0, pan2_3)
        pan2_4 = self.lrelu(self.conv_pan2_4(pan2_add_1))  # 256x32x32
        pan2_5 = self.lrelu(self.conv_pan2_5(pan2_4))
        pan2_6 = self.lrelu(self.conv_pan2_6(pan2_5))
        pan2_7 = self.conv_pan2_7(pan2_6)
        F_hsv22 = torch.add(pan2_4, pan2_7)
        F_hsv2 = torch.max_pool2d(F_hsv22, kernel_size=2, stride=2, padding=0)
        # third encoder
        pan3_0 = self.lrelu(self.conv_pan3_0(F_hsv2))
        pan3_1 = self.lrelu(self.conv_pan3_1(pan3_0))
        pan3_2 = self.lrelu(self.conv_pan3_2(pan3_1))
        pan3_3 = self.conv_pan3_3(pan3_2)
        pan3_add_1 = torch.add(pan3_0, pan3_3)
        pan3_4 = self.lrelu(self.conv_pan3_4(pan3_add_1))
        pan3_5 = self.lrelu(self.conv_pan3_5(pan3_4))
        pan3_6 = self.lrelu(self.conv_pan3_6(pan3_5))
        pan3_7 = self.conv_pan3_7(pan3_6)
        F_hsv3 = torch.add(pan3_4, pan3_7)
        ########
        ##RGB  first
        ms1_0 = self.lrelu(self.conv_ms1_0(rgb))
        ms1_1 = self.lrelu(self.conv_ms1_1(ms1_0)) # 256*3=768===>256
        ms1_2 = self.lrelu(self.conv_ms1_2(ms1_1))  # 768==>256
        ms1_3 = self.conv_ms1_3(ms1_2)  # 768===>256
        ms1_add_1 = torch.add(ms1_0, ms1_3)  # 256
        ms1_4 = self.lrelu(self.conv_ms1_4(ms1_add_1))  # 256x32x32
        ms1_5 = self.lrelu(self.conv_ms1_5(ms1_4))  # 256*3=768===>256
        ms1_6 = self.lrelu(self.conv_ms1_6(ms1_5))  # 768==>256
        ms1_7 = self.conv_ms1_7(ms1_6)
        F_rgb11= torch.add(ms1_4,  ms1_7)
        F_rgb1 = torch.max_pool2d(F_rgb11, kernel_size=2, stride=2, padding=0)# 256
        #second
        ms2_0 = self.lrelu(self.conv_ms2_0(F_rgb1))  #
        ms2_1 = self.lrelu(self.conv_ms2_1(ms2_0))
        ms2_2 = self.lrelu(self.conv_ms2_2(ms2_1))
        ms2_3 = self.conv_ms2_3(ms2_2)
        ms2_add_1 = torch.add(ms2_0, ms2_3)  # 128
        ms2_4 = self.lrelu(self.conv_ms2_4(ms2_add_1))
        ms2_5 = self.lrelu(self.conv_ms2_5(ms2_4))
        ms2_6 = self.lrelu(self.conv_ms2_6(ms2_5))
        ms2_7 = self.conv_ms2_7(ms2_6)
        F_rgb22 = torch.add(ms2_4, ms2_7)
        F_rgb2 = torch.max_pool2d(F_rgb22, kernel_size=2, stride=2, padding=0)
        # third
        ms3_0 = self.lrelu(self.conv_ms3_0(F_rgb2))
        ms3_1 = self.lrelu(self.conv_ms3_1(ms3_0))
        ms3_2 = self.lrelu(self.conv_ms3_2(ms3_1))
        ms3_3 = self.conv_ms3_3(ms3_2)
        ms3_add_1 = torch.add(ms3_0, ms3_3)
        ms3_4 = self.lrelu(self.conv_ms3_4(ms3_add_1))
        ms3_5 = self.lrelu(self.conv_ms3_5(ms3_4))
        ms3_6 = self.lrelu(self.conv_ms3_6(ms3_5))
        ms3_7 = self.conv_ms3_7(ms3_6)
        F_rgb3 = torch.add(ms3_4, ms3_7)

        ####lab  first
        hs1_0 = self.lrelu(self.conv_hs1_0(lab))  #
        hs1_1 = self.lrelu(self.conv_hs1_1(hs1_0))
        hs1_2 = self.lrelu(self.conv_hs1_2(hs1_1))
        hs1_3 = self.conv_hs1_3(hs1_2)
        hs1_add_1 = torch.add(hs1_0, hs1_3)  # 128
        hs1_4 = self.lrelu(self.conv_hs1_4(hs1_add_1))
        hs1_5 = self.lrelu(self.conv_hs1_5(hs1_4))
        hs1_6 = self.lrelu(self.conv_hs1_6(hs1_5))
        hs1_7 = self.conv_hs1_7(hs1_6)
        F_lab11 = torch.add(hs1_4, hs1_7)
        F_lab1 = torch.max_pool2d(F_lab11, kernel_size=2, stride=2, padding=0)
        # second
        hs2_0 = self.lrelu(self.conv_hs2_0(F_lab1))
        hs2_1 = self.lrelu(self.conv_hs2_1(hs2_0))
        hs2_2 = self.lrelu(self.conv_hs2_2(hs2_1))
        hs2_3 = self.conv_hs2_3(hs2_2)
        hs2_add_1 = torch.add(hs2_0, hs2_3)
        hs2_4 = self.lrelu(self.conv_hs2_4(hs2_add_1))
        hs2_5 = self.lrelu(self.conv_hs2_5(hs2_4))
        hs2_6 = self.lrelu(self.conv_hs2_6(hs2_5))
        hs2_7 = self.conv_hs2_7(hs2_6)
        F_lab22 = torch.add(hs2_4, hs2_7)
        F_lab2 = torch.max_pool2d(F_lab22, kernel_size=2, stride=2, padding=0)
        # third
        hs3_0 = self.lrelu(self.conv_hs3_0(F_lab2))
        hs3_1 = self.lrelu(self.conv_hs3_1(hs3_0))
        hs3_2 = self.lrelu(self.conv_hs3_2(hs3_1))
        hs3_3 = self.conv_hs3_3(hs3_2)
        hs3_add_1 = torch.add(hs3_0, hs3_3)
        hs3_4 = self.lrelu(self.conv_hs3_4(hs3_add_1))
        hs3_5 = self.lrelu(self.conv_hs3_5(hs3_4))
        hs3_6 = self.lrelu(self.conv_hs3_6(hs3_5))
        hs3_7 = self.conv_hs3_7(hs3_6)
        F_lab3 = torch.add(hs3_4, hs3_7)

        ######concate
        first_cat = torch.cat((F_hsv11, F_rgb11, F_lab11), dim=1)
        first_cat_se = self.lrelu(torch.sigmoid(self.se_1(first_cat)) * first_cat) + first_cat
        f3 = self.lrelu(self.conv_cat_1(first_cat_se))  # 384===>128

        # second layer
        second_cat = torch.cat((F_hsv22, F_rgb22, F_lab22), dim=1)
        second_cat_se = self.lrelu(torch.sigmoid(self.se_2(second_cat)) * second_cat) + second_cat
        f2 = self.lrelu(self.conv_cat_2(second_cat_se))  # 768===>256

        # third layer
        third_cat = torch.cat((F_hsv3, F_rgb3, F_lab3), dim=1)
        third_cat_se = self.lrelu(torch.sigmoid(self.se_3(third_cat)) * third_cat) + third_cat
        f1 = self.lrelu(self.conv_cat_3(third_cat_se))  # 1536===>512

        decoder_input1 = torch.add(f1, torch.multiply(f1, depth_4down))
        decoder_input1_1 = self.lrelu(self.conv_de1_0(decoder_input1))
        decoder_input1_2 = self.lrelu(self.conv_de1_1(decoder_input1_1))
        decoder_input1_3 = self.lrelu(self.conv_de1_2(decoder_input1_2))
        decoder_input1_4 = self.conv_de1_3(decoder_input1_3)
        decoder_input1_5 = torch.add(decoder_input1_4, decoder_input1_1)
        decoder_input1_6 = self.lrelu(self.conv_de1_4(decoder_input1_5))
        decoder_input1_7 = self.lrelu(self.conv_de1_5(decoder_input1_6))
        decoder_input1_8 = self.lrelu(self.conv_de1_6(decoder_input1_7))
        decoder_input1_9 = self.conv_de1_7(decoder_input1_8)
        decoder_input1_10 = torch.add(decoder_input1_9, decoder_input1_6)
        lay3 = nn.functional.interpolate(decoder_input1_10, scale_factor=2, mode='bilinear', align_corners=False)

        decoder_input2 = torch.add(f2, torch.multiply(f2, depth_2down))
        decoder_input2 = torch.cat((decoder_input2,lay3),dim=1)
        decoder_input2_1 = self.lrelu(self.conv_de2_0(decoder_input2))
        decoder_input2_2 = self.lrelu(self.conv_de2_1(decoder_input2_1))
        decoder_input2_3 = self.lrelu(self.conv_de2_2(decoder_input2_2))
        decoder_input2_4 = self.conv_de2_3(decoder_input2_3)
        decoder_input2_5 = torch.add(decoder_input2_4, decoder_input2_1)
        decoder_input2_6 = self.lrelu(self.conv_de2_4(decoder_input2_5))
        decoder_input2_7 = self.lrelu(self.conv_de2_5(decoder_input2_6))
        decoder_input2_8 = self.lrelu(self.conv_de2_6(decoder_input2_7))
        decoder_input2_9 = self.conv_de2_7(decoder_input2_8)
        decoder_input2_10 = torch.add(decoder_input2_9, decoder_input2_6)
        lay2 = nn.functional.interpolate(decoder_input2_10, scale_factor=2, mode='bilinear', align_corners=False)


        decoder_input3 = torch.add(f3, torch.multiply(f3, 1 - _depth))
        decoder_input3 = torch.cat((decoder_input3, lay2), dim=1)
        decoder_input3_1 = self.lrelu(self.conv_de3_0(decoder_input3))
        decoder_input3_2 = self.lrelu(self.conv_de3_1(decoder_input3_1))
        decoder_input3_3 = self.lrelu(self.conv_de3_2(decoder_input3_2))
        decoder_input3_4 = self.conv_de3_3(decoder_input3_3)
        decoder_input3_5 = torch.add(decoder_input3_4, decoder_input3_1)
        decoder_input3_6 = self.lrelu(self.conv_de3_4(decoder_input3_5))
        decoder_input3_7 = self.lrelu(self.conv_de3_5(decoder_input3_6))
        decoder_input3_8 = self.lrelu(self.conv_de3_6(decoder_input3_7))
        decoder_input3_9 = self.conv_de3_7(decoder_input3_8)
        decoder_input3_10 = torch.add(decoder_input3_9, decoder_input3_6)
        out = self.conv_de4(decoder_input3_10)

        result = torch.clamp(out, min=0, max=1)

        return result

if __name__ == '__main__':
    test = torch.randn(1, 3, 512, 512).cuda()
    model = Model().cuda()
    res = model(test, test)
    print(res.shape)
