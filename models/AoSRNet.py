import torch
import torch.nn as nn
import torch.nn.functional as F


class AoSRNet(nn.Module):
	def __init__(self):
		super(AoSRNet,self).__init__()

		self.mns = MainNetworkStructure(3,12)
         
	def forward(self,x):
        
		Fout = self.mns(x)
      
		return Fout# + x


class MainNetworkStructure(nn.Module):
	def __init__(self,inchannel,channel):
		super(MainNetworkStructure,self).__init__()

		self.conv_mv_in = nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1,dilation=1,bias=False)
		self.conv_wb_in = nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1,dilation=1,bias=False)
		self.conv_gc_in = nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1,dilation=1,bias=False)
        
        
		self.conv_mv = BB(channel)   
		self.conv_wb = BB(channel)
		self.conv_gc = BB(channel)
        
		self.ED = En_Decoder(channel,3*channel) 
    		
		self.wbm = WBM()
		self.gcm = GCM()
		self.mvp = MVP()
        
	def forward(self,x):

		mv_x1 = torch.clamp(self.conv_mv_in(x),1e-10,1.0)
		mv_x2 = torch.clamp(self.conv_mv_in(x),1e-10,1.0)
		mv_x3 = torch.clamp(self.conv_mv_in(x),1e-10,1.0)
		mv_x4 = torch.clamp(self.conv_mv_in(x),1e-10,1.0)
        
		wb_x1 = torch.clamp(self.conv_wb_in(x),1e-10,1.0)
		wb_x2 = torch.clamp(self.conv_wb_in(x),1e-10,1.0)
		wb_x3 = torch.clamp(self.conv_wb_in(x),1e-10,1.0)
		wb_x4 = torch.clamp(self.conv_wb_in(x),1e-10,1.0)

		gc_x1 = torch.clamp(self.conv_gc_in(x),1e-10,1.0)
		gc_x2 = torch.clamp(self.conv_gc_in(x),1e-10,1.0)
		gc_x3 = torch.clamp(self.conv_gc_in(x),1e-10,1.0)
		gc_x4 = torch.clamp(self.conv_gc_in(x),1e-10,1.0)


		mv = self.conv_mv(self.mvp(mv_x1,mv_x2,mv_x3,mv_x4))        
		wb = self.conv_wb(self.wbm(wb_x1,wb_x2,wb_x3,wb_x4))  
		gc = self.conv_gc(self.gcm(gc_x1,gc_x2,gc_x3,gc_x4))       

        
		x_out = self.ED(mv,wb,gc)
        
		return x_out# + x

    
class MVP(nn.Module):    #Multi-view perception
	def __init__(self,norm=False):                                
		super(MVP,self).__init__()

		self.convD_3  = nn.Conv2d(3,3,kernel_size=3,stride=1,padding=3,dilation=3,bias=False)
		self.convD_6  = nn.Conv2d(3,3,kernel_size=3,stride=1,padding=6,dilation=6,bias=False)
		self.convD_9  = nn.Conv2d(3,3,kernel_size=3,stride=1,padding=9,dilation=9,bias=False)
		self.convD_12 = nn.Conv2d(3,3,kernel_size=3,stride=1,padding=12,dilation=12,bias=False)

		self.act = nn.PReLU(3)
		self.norm = nn.GroupNorm(num_channels=3,num_groups=1)# nn.InstanceNorm2d(channel)#
   
	def forward(self,x1,x2,x3,x4):
        
		x1 = self.act(self.norm(self.convD_3(x1)))
		x2 = self.act(self.norm(self.convD_6(x2)))      
		x3 = self.act(self.norm(self.convD_9(x3)))
		x4 = self.act(self.norm(self.convD_12(x4)))          
        
		xout = torch.cat((x1,x2,x3,x4),1)  
        
		return	xout
    
class WBM(nn.Module):    # White Balance Model
	def __init__(self):
		super(WBM,self).__init__()

		self.conv_1 = ConvL(3,3)
		self.conv_2 = ConvL(3,3)
		self.conv_3 = ConvL(3,3)
		self.conv_4 = ConvL(3,3)

	def forward(self,x1,x2,x3,x4):    

		x1 = self.conv_1(WhiteBalance(x1,0.05,0.10))
		x2 = self.conv_2(WhiteBalance(x2,0.05,0.15))
		x3 = self.conv_3(WhiteBalance(x3,0.15,0.10))
		x4 = self.conv_4(WhiteBalance(x4,0.15,0.20))

		xout = torch.cat((x1,x2,x3,x4),1)   
                       
		return xout


def WhiteBalance(TensorData,pmi,pma):   
	'''White Balance for recovery priors'''  
	for i in range(TensorData.shape[0]):
		for j in range(3):
			tmi  = torch.quantile(TensorData[i,j,:,:].clone(),0.01)
			tma  = torch.quantile(TensorData[i,j,:,:].clone(),0.09) 
			tpmi = tmi - pmi * (tma - tmi) 			
			tpma = tma + pma * (tma - tmi) 
            
			TensorData[i,j,:,:,] = (TensorData[i,j,:,:].clone() - tpmi) / ((tpma - tpmi) + 1e-10) 
     
	return TensorData

    
class GCM(nn.Module):    # Gamma Correction Model
	def __init__(self):
		super(GCM,self).__init__()
        
		self.conv_1 = ConvL(3,3)
		self.conv_2 = ConvL(3,3)
		self.conv_3 = ConvL(3,3)
		self.conv_4 = ConvL(3,3)
        
	def forward(self,x1,x2,x3,x4):    

		x1 = self.conv_1(torch.pow(x1,1/4))
		x2 = self.conv_2(torch.pow(x2,1/2))
		x3 = self.conv_3(torch.pow(x3,2))
		x4 = self.conv_4(torch.pow(x4,4))

		xout = torch.cat((x1,x2,x3,x4),1)

		return xout
    
class BB(nn.Module):    #Basic Block (BB)
	def __init__(self,channel,norm=False):                                
		super(BB,self).__init__()

		self.conv_1 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_2 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)   
		self.conv_3 = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)  
		self.conv_out = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.act = nn.PReLU(channel)
		self.norm = nn.GroupNorm(num_channels=channel,num_groups=1)# nn.InstanceNorm2d(channel)#
   
	def forward(self,x):
        
		x_1 = self.act(self.norm(self.conv_1(x)))
		x_2 = self.act(self.norm(self.conv_2(x_1)))
		x_out = self.act(self.norm(self.conv_out(x_2)) + x)

		return	x_out

    
class ConvL(nn.Module):
	def __init__(self,inchannel,channel,norm=False):                                
		super(ConvL,self).__init__()

		self.conv = nn.Conv2d(inchannel,channel,kernel_size=3,stride=1,padding=1,bias=False)
		self.act = nn.PReLU(channel)
		self.norm = nn.GroupNorm(num_channels=channel,num_groups=1)
   
	def forward(self,x):
        
		x_out = self.act(self.norm(self.conv(x)))

		return	x_out      


class En_Decoder(nn.Module):
	def __init__(self,inchannel,channel):
		super(En_Decoder,self).__init__()

		self.el = BB(channel)
		self.em = BB(channel*2)
		self.es = BB(channel*4)
		self.ds = BB(channel*4)
		self.dm = BB(channel*2)
		self.dl = BB(channel)

		self.conv_eltem = nn.Conv2d(channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_emtes = nn.Conv2d(2*channel,4*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_dstdm = nn.Conv2d(4*channel,2*channel,kernel_size=1,stride=1,padding=0,bias=False)   
		self.conv_dmtdl = nn.Conv2d(2*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)  

		self.conv_in = nn.Conv2d(12,channel,kernel_size=3,stride=1,padding=1,bias=False)
		#self.conv_cat_in = nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_out = nn.Conv2d(channel,3,kernel_size=3,stride=1,padding=1,bias=False)    
		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

	def _upsample(self,x,y):
		_,_,H,W = y.size()
		return F.interpolate(x,size=(H,W),mode='bilinear')

	def forward(self,x1,x2,x3):
             
		x_elin = torch.cat((x1,x2,x3),1) + self.conv_in(x1+x2+x3)# + self.conv_in(x1)
        
		elout = self.el(x_elin)        
		emout = self.em(self.conv_eltem(self.maxpool(elout)))        
		esout = self.es(self.conv_emtes(self.maxpool(emout)))
        
		dsout = self.ds(esout)
		dmout = self.dm(self._upsample(self.conv_dstdm(dsout),emout) + emout)
		dlout = self.dl(self._upsample(self.conv_dmtdl(dmout),elout) + elout)

		x_out = self.conv_out(dlout)

		return x_out
	

if __name__ == '__main__':
	inp = torch.randn(1, 3, 256, 256).cuda()
	model = AoSRNet().cuda()
	res = model(inp)
	print(res.shape)
	