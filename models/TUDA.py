import torch
import torch.nn as nn

# 4个 conv 的 Dense block
class Dense_Block_IN(nn.Module):
    def __init__(self, block_num, inter_channel, channel):
        super(Dense_Block_IN, self).__init__()
        #
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
        #
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_channels, channel, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(channel, affine=True),
            nn.ReLU(),
        )
        #

    def forward(self, x):
        feature_list = [x]

        for group in self.group_list:
            inputs = torch.cat(feature_list, dim=1)
            outputs = group(inputs)
            feature_list.append(outputs)

        inputs = torch.cat(feature_list, dim=1)
        #
        fusion_outputs = self.fusion(inputs)
        #
        block_outputs = fusion_outputs + x

        return block_outputs

class CALayer(nn.Module):
	def __init__(self, channel):
		super(CALayer, self).__init__()
		out_channel = channel
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
		#
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.ca = nn.Sequential(
				nn.Conv2d(out_channel, out_channel // 8, 1, padding=0, bias=True),
				nn.ReLU(inplace=True),
				nn.Conv2d(out_channel // 8, channel, 1, padding=0, bias=True),
				nn.Sigmoid()
		)

	def forward(self, x):
		t1 = self.conv1(x)  # in
		t2 = self.relu(t1)  # in, 64
		y = self.avg_pool(t2)  # torch.Size([1, in, 1, 1])
		y = self.ca(y)  # torch.Size([1, in, 1, 1])
		m = t2 * y      # torch.Size([1, in, 64, 64]) * torch.Size([1, in, 1, 1])
		return x + m


class PALayer(nn.Module):
	def __init__(self, channel):
		super(PALayer, self).__init__()
		self.pa = nn.Sequential(
				nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
				nn.ReLU(inplace=True),
				nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
				nn.Sigmoid()
		)

	def forward(self, x):
		y = self.pa(x)
		return x * y
	
# upsample
class Trans_Up(nn.Module):
	def __init__(self, in_planes, out_planes):
		super(Trans_Up, self).__init__()
		self.conv0 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1)
		self.IN1 = nn.InstanceNorm2d(out_planes)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.relu(self.IN1(self.conv0(x)))
		return out


# downsample
class Trans_Down(nn.Module):
	def __init__(self, in_planes, out_planes):
		super(Trans_Down, self).__init__()
		self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1)
		self.IN1 = nn.InstanceNorm2d(out_planes)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.relu(self.IN1(self.conv0(x)))
		return out
	
class TUDA(nn.Module):
	def __init__(self, input_nc=3, output_nc=3):
		super(TUDA, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(input_nc, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		#
		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		# 几个 conv, 中间 channel, 输入 channel
		self.up_1 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 256 -> 128
		self.up_2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 128 -> 64
		self.up_3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 64 -> 32
		self.up_4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 32 -> 16
		#
		self.Latent = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 16 -> 8
		#
		self.down_4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 16
		self.down_3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 32
		self.down_2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 64
		self.down_1 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 128
		#
		self.CALayer4 = CALayer(128)
		self.CALayer3 = CALayer(128)
		self.CALayer2 = CALayer(128)
		self.CALayer1 = CALayer(128)
		#
		self.trans_down1 = Trans_Down(64, 64)
		self.trans_down2 = Trans_Down(64, 64)
		self.trans_down3 = Trans_Down(64, 64)
		self.trans_down4 = Trans_Down(64, 64)
		#
		self.trans_up4 = Trans_Up(64, 64)
		self.trans_up3 = Trans_Up(64, 64)
		self.trans_up2 = Trans_Up(64, 64)
		self.trans_up1 = Trans_Up(64, 64)
		#
		self.down_4_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.down_3_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.down_2_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.down_1_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		#
		self.fusion = nn.Sequential(
			nn.Conv2d(64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU()
		)
		self.fusion2 = nn.Sequential(
			nn.Conv2d(64, output_nc, 3, 1, 1),
			nn.Tanh(),
		)
		#

	def forward(self, x):   # 1, 3, 256, 256
		#
		feature_neg_1 = self.conv1(x)   # 1, 64, 256, 256
		feature_0 = self.conv2(feature_neg_1)   # # 1, 64, 256, 256
		#######################################################
		up_11 = self.up_1(feature_0)  # 1, 64, 256, 256
		up_1 = self.trans_down1(up_11)   # 1, 64, 128, 128

		up_21 = self.up_2(up_1)  # 1, 64, 128, 128
		up_2 = self.trans_down2(up_21)    # 1, 64, 64, 64

		up_31 = self.up_3(up_2)  # 1, 64, 64, 64
		up_3 = self.trans_down3(up_31)    # 1, 64, 32, 32

		up_41 = self.up_4(up_3)  # 1, 64, 32, 32
		up_4 = self.trans_down4(up_41)    # 1, 64, 16, 16

		#######################################################
		Latent = self.Latent(up_4)  # 1, 64, 16, 16
		#######################################################

		down_4 = self.trans_up4(Latent)  # 1, 64, 32, 32
		down_4 = torch.cat([up_41, down_4], dim=1)  # 1, 128, 32, 32
		down_41 = self.CALayer4(down_4)     # 1, 128, 32, 32
		down_4 = self.down_4_fusion(down_41)     # 1, 64, 32, 32
		down_4 = self.down_4(down_4)       # 1, 64, 32, 32

		down_3 = self.trans_up3(down_4)  # 1, 64, 64, 64
		down_3 = torch.cat([up_31, down_3], dim=1)  # 1, 128, 64, 64
		down_31 = self.CALayer3(down_3)     # 1, 128, 64, 64
		down_3 = self.down_3_fusion(down_31)     # 1, 64, 64, 64
		down_3 = self.down_3(down_3)       # 1, 64, 64, 64

		down_2 = self.trans_up2(down_3)  # 1, 64, 128,128
		down_2 = torch.cat([up_21, down_2], dim=1)  # 1, 128, 128,128
		down_21 = self.CALayer2(down_2)     # 1, 128, 128,128
		down_2 = self.down_2_fusion(down_21)     # 1, 64, 128,128
		down_2 = self.down_2(down_2)       # 1, 64, 128,128

		down_1 = self.trans_up1(down_2)  # 1, 64, 256, 256
		down_1 = torch.cat([up_11, down_1], dim=1)  # 1, 128, 256, 256
		down_11 = self.CALayer1(down_1)     # 1, 128, 256, 256
		down_1 = self.down_1_fusion(down_11)     # 1, 64, 256, 256
		down_1 = self.down_1(down_1)       # 1, 64, 256, 256
		#
		feature = self.fusion(down_1)  # 1, 64, 256, 256
		#
		feature = feature + feature_neg_1  # 1, 64, 256, 256
		#
		outputs = self.fusion2(feature)
		return outputs
	

if __name__ == '__main__':
	t = torch.randn(1, 3, 256, 256).cuda()
	model = TUDA().cuda()
	res = model(t)
	print(res.shape)