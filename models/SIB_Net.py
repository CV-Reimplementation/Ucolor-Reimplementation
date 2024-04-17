import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

class conv_Ucolor(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch,kernelsize=3,padding=1,stride=1):
        super(conv_Ucolor, self).__init__()
        self.kernelsize=kernelsize
        self.padding=padding
        self.stride=stride
        self.conv1 =nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=self.kernelsize, padding=self.padding,stride=self.stride),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(0.2),
            nn.ReLU()
        )
    def forward(self,x):
        out=self.conv1(x)
        return out


class residualBlock(nn.Module):
    """
    由开头的MulitiColorblock和后面的Multicolorblock_inter组成
    """

    def __init__(self, in_ch=3, first_ch=64,out_ch=64,is_relu=True):
        super(residualBlock, self).__init__()
        self.in_ch=in_ch
        self.first_ch=first_ch
        self.out_ch=out_ch
        self.is_relu=is_relu
        self.conv1=nn.Sequential(
            nn.Conv2d(self.in_ch, self.first_ch, 3, 1, padding=1),
            nn.BatchNorm2d(self.first_ch),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(self.first_ch, self.out_ch, 3, 1, padding=1),
            nn.BatchNorm2d(self.out_ch),
            nn.Dropout(0.2)
        )
        if self.is_relu:
            self.relu = nn.Sequential(nn.ReLU())

        if(self.in_ch!=self.out_ch):
            self.conv_cross=nn.Sequential(
                nn.Conv2d(self.in_ch,self.out_ch,1,1),
                nn.BatchNorm2d(self.out_ch)
            )

    def forward(self, x):

        if self.is_relu:
            if (self.in_ch != self.out_ch):
                inter = self.conv1(x)
                result = self.relu(self.conv2(inter) + self.conv_cross(x))
            else:
                inter = self.conv1(x)
                result = self.relu(x + self.conv2(inter))
        else:
            if (self.in_ch != self.out_ch):
                inter = self.conv1(x)
                result = self.conv2(inter) + self.conv_cross(x)
            else:
                inter = self.conv1(x)
                result =x + self.conv2(inter)

        return result




class ColorBlock(nn.Module):
    """
    由开头的MulitiColorblock和后面的Multicolorblock_inter组成
    """

    def __init__(self,in1=3,in2=64,in3=64,in4=64,in5=64,in6=128,is_pool=True,):
        super(ColorBlock, self).__init__()

        self.in1=in1
        self.in2=in2
        self.in3=in3
        self.in4=in4
        self.in5=in5
        self.in6=in6
        self.is_pool=is_pool


        #RGB
        self.rgb1=residualBlock(self.in1,self.in2,self.in3)
        self.rgb2 = residualBlock(self.in4,self.in5,self.in6)

        self.pool=nn.MaxPool2d(2)

    def forward(self,x):

        rgb1=self.rgb1(x)   #64维度
        rgb2=self.rgb2(rgb1)  #128,H,W

        if(self.is_pool):
            rgb_down = self.pool(rgb2)
        else:
            rgb_down=rgb2

        return rgb2,rgb_down




class MultiColor(nn.Module):
    """
    由开头的MulitiColorblock和后面的Multicolorblock_inter组成
    """

    def __init__(self):
        super(MultiColor, self).__init__()
        self.block1=ColorBlock()
        self.block2=ColorBlock(128,128,128,128,256,256)
        self.block3=ColorBlock(256,256,256,256,512,512,is_pool=False)


    def forward(self,x):

        #第一层
        cat1,cat1_down=self.block1(x)  # 128,256,256   down:128,128,128

        #第二层
        cat2,cat2_down=self.block2(cat1_down) # 256*3,128,128  dowmn:256,64,64

        #第三层
        cat3,_=self.block3(cat2_down)  #512,64,64



        return cat3,cat2,cat1


# model=MultiColor()
# x=torch.randn(1,3,256,256)
# model=model.cuda()
# x=x.cuda()
# a,b,c=model(x)
# print(a.shape)
# print(b.shape)
# print(c.shape)

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
class M_block(nn.Module):
    """
    就是原文里的注意力通道机制的那个模块
    """
    def __init__(self, in_ch, out_ch,ratio=16):
        super(M_block, self).__init__()
        self.Globalavg=nn.AdaptiveAvgPool2d(1)
        #reshape一下
        self.dense_conect=nn.Sequential(
            nn.Linear(in_ch,int(in_ch/ratio)),
            nn.ReLU(),
            nn.Linear(int(in_ch/ratio),in_ch),
            nn.Sigmoid()
        )
        #reshape回去
        self.conv=nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        """
        input是多个颜色空间合起来的特征
        """

        x1=self.Globalavg(x)   # (3,channel,H,W)  to (3,channel,1,1)
        x2=torch.reshape(x1,x1.shape[0:2])   # (3,channel)
        x3=self.dense_conect(x2)
        x4=torch.reshape(x3,[x3.shape[0],x3.shape[1],1,1])  #(3,channel,1,1)
        x5=x4*x
        out=self.conv(x5)

        return out


class Mblock(nn.Module):
    """
    接受三个输入,产生三个输出
    """
    def __init__(self, in_ch=[512,256,128], out_ch=[512,256,128]):
        super(Mblock, self).__init__()
        self.block1=M_block(in_ch[0],out_ch[0])  # 对应output512
        self.block2=M_block(in_ch[1],out_ch[1])  # 对应output256
        self.block3=M_block(in_ch[2], out_ch[2])  # 对应output128

    def forward(self,out512,out256,out128):
        channel_out512=self.block1(out512)
        channel_out256 = self.block2(out256)
        channel_out128=self.block3(out128)
        return channel_out512,channel_out256,channel_out128


class spatial_attention_block(nn.Module):
    def __init__(self):
        super(spatial_attention_block, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out    #

class spatial_attention(nn.Module):
    """
    接受三个输入,产生三个输出
    """
    def __init__(self):
        super(spatial_attention, self).__init__()
        self.block1=spatial_attention_block()  # 对应output512
        self.block2=spatial_attention_block()  # 对应output256
        self.block3=spatial_attention_block()  # 对应output128

    def forward(self,out512,out256,out128):
        channel_out512=self.block1(out512)
        channel_out256 = self.block2(out256)
        channel_out128=self.block3(out128)
        return channel_out512,channel_out256,channel_out128




#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

class semantic_Decoder(nn.Module):
    """
    接受三个输入,产生三个输出
    """
    def __init__(self):
        super(semantic_Decoder, self).__init__()
        self.block1=residualBlock(512,256,256)
        self.up1=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True) #256,128,128
        self.block2=residualBlock(512,256,128) #拼接一下,再卷回128,128,128
        self.up2=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True) #128,256,256
        self.block3=residualBlock(256,128,128)
        self.out_block=nn.Sequential(
            residualBlock(128,64,64),
            conv_Ucolor(64,32),
            nn.Conv2d(32,3,3,1,1)
        )
        self.fusion_block=residualBlock(128,128,128)



    def forward(self,out512,out256,out128):
        out=self.up1(self.block1(out512))
        out=self.up2(self.block2(torch.cat([out,out256],dim=1)))
        out=self.block3(torch.cat([out,out128],dim=1))
        As=self.out_block(out)
        As_feature=self.fusion_block(out)
        return As,As_feature


class A_Decoder(nn.Module):
    """
    接受三个输入,产生三个输出
    """

    def __init__(self):
        super(A_Decoder, self).__init__()
        self.block1 = residualBlock(512, 256, 256)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 256,128,128
        self.block2 = residualBlock(512, 256, 128)  # 拼接一下,再卷回128,128,128
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 128,256,256
        self.block3 = residualBlock(256, 128, 128)# 拼接128，再卷回128
        self.semantic_fusion_block=nn.Sequential(
            residualBlock(256,128,128),
            #residualBlock(128,128,128),
            residualBlock(128, 64, 32),
            #conv_Ucolor(64, 32),
            nn.Conv2d(32, 3, 3, 1, 1)
        )


    def forward(self, out512, out256, out128,semantic_feature):
        out = self.up1(self.block1(out512))
        out = self.up2(self.block2(torch.cat([out, out256], dim=1)))
        out = self.block3(torch.cat([out, out128], dim=1))
        out=self.semantic_fusion_block(torch.cat([out,semantic_feature],dim=1))

        return out





class Grad_Net(nn.Module):
    """
    接受三个输入,产生三个输出
    """

    def __init__(self):
        super(Grad_Net, self).__init__()
        self.block1=conv_Ucolor(3,32)
        #self.block2=residualBlock(32,32,32)
        self.feature_con1=nn.Conv2d(128,32,1)

        self.block3=nn.Sequential(  # 和第一个过来的拼接然后输入，并且输入还要连到后面
            residualBlock(64,64,64),
            nn.MaxPool2d(2)
        )
        self.feature_con2 = nn.Conv2d(256, 64, 1)   # 和第二个过来的拼接
        self.block4=nn.Sequential(
            #conv_Ucolor(128,128),
            residualBlock(128,64,64)
        )
        self.block5= conv_Ucolor(128,64) #和前面的拼接
        self.up1 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.block6=residualBlock(64,64,64)   #送出去连接的
        self.block7=conv_Ucolor(64,32)               #和block2相连的
        self.block8=residualBlock(64,32,32)
        self.outblock=nn.Conv2d(32,3,1)


    def forward(self, x_grad, out256, out128 ):
        out256=self.feature_con2(out256)
        out128=self.feature_con1(out128)
        x1=self.block1(x_grad)
        x2=self.block3(torch.cat([x1,out128],dim=1))  # 64 128
        x3=self.block4(torch.cat([x2,out256],dim=1))  #64
        x3=self.up1(self.block5(torch.cat([x3,x2],dim=1)))   #送出去的64
        x4=self.block8(torch.cat([self.block7(x3),x1],dim=1))  #送出去的32

        out=self.outblock(x4)

        return out,x3



class T_Decoder(nn.Module):
    """
    接受三个输入,产生三个输出
    """

    def __init__(self):
        super(T_Decoder, self).__init__()
        self.block1 = residualBlock(512, 256, 256)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 256,128,128
        self.block2 = residualBlock(512, 256, 128)  # 拼接一下,再卷回128,128,128
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 128,256,256
        self.block3 = residualBlock(256, 128, 128)  # 拼接128，再卷回128
        self.out_block1=nn.Sequential(
            #residualBlock(128, 128, 128),
            residualBlock(128, 64, 64)
        )
        self.out_block2 = residualBlock(128, 64, 64)    # 和64拼接一下

        self.out_block3=nn.Sequential(
            #residualBlock(64, 64, 64),
            conv_Ucolor(64, 32),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )


    def forward(self, out512, out256, out128 ,grad64 ):
        out = self.up1(self.block1(out512))
        out = self.up2(self.block2(torch.cat([out, out256], dim=1)))
        out = self.block3(torch.cat([out, out128], dim=1))
        out = self.out_block1(out)
        out = self.out_block3(self.out_block2(torch.cat([out,grad64],dim=1)))

        return out


class SIB_Net(nn.Module):
    """
    input：rgb图（b,3,H,W）和深度图（b,1,H,W）
    output:最终结果（b,3,H,W）
    """

    def __init__(self, in_ch=[512, 256, 128], out_ch=3, ratio=16):
        super(SIB_Net, self).__init__()
        self.in_ch=in_ch
        self.color_embedding=MultiColor()
        self.Mblock=Mblock([self.in_ch[0],self.in_ch[1],self.in_ch[2]], out_ch=self.in_ch)
        self.spatial_att=spatial_attention()
        self.semantic_decoder=semantic_Decoder()
        self.SS_decoder=A_Decoder()
        self.GS_decoder=T_Decoder()
        self.grad_net=Grad_Net()
        self.persudo1=nn.Sequential(
            residualBlock(256, 64, 32),
            conv_Ucolor(32, 32),
        )
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True) #256,128,128
        self.persudo2= nn.Conv2d(32, 3, 3, 1, 1)
        self.final_conv=nn.Conv2d(9,3,1)
        self.lambda1=nn.Parameter(torch.FloatTensor(torch.ones(256,256)))
        self.lambda2 = nn.Parameter(torch.FloatTensor(torch.ones( 256, 256)))
        self.lambda3 = nn.Parameter(torch.FloatTensor(torch.ones( 256, 256)))

    def forward(self,x, x_e, x_s):
        x512_pre,x256_pre,x128_pre=self.color_embedding(x)

        y512,y256,y128=self.Mblock(x512_pre,x256_pre,x128_pre)
        s5,s2,s1=self.spatial_att(x512_pre,x256_pre,x128_pre)
        s512= x512_pre*s5
        s256 = x256_pre * s2
        s128 = x128_pre * s1
        grad_out,grad64=self.grad_net(x_e,y256,y128)
        As,As_feature=self.semantic_decoder(s512,s256,s128)
        out1=self.SS_decoder(y512,y256,y128,As_feature)
        out2=self.GS_decoder(y512,y256,y128,grad64)
        pixel_out=self.persudo2(self.up(self.persudo1(x256_pre)))
        out=self.final_conv(torch.cat([out1*x_s,out2,pixel_out],dim=1))



        return out
    

if __name__ == '__main__':
    inp = torch.randn(1, 3, 256, 256).cuda()
    model = SIB_Net().cuda()
    res = model(inp, inp, inp)
    print(res.shape)