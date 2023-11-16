import torch
import torch.nn as nn
from torch.nn import ModuleList
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import numpy as np
import copy
from torch.nn.functional import conv2d, conv_transpose2d
import math
import torch.nn.functional as F 


#线性编码
class Channel_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(0.1)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # (B, hidden，n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


#特征重组
class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        B, n_patch, hidden = x.size()  
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class Attention_org(nn.Module):
    def __init__(self, vis,channel_num, KV_size=480, num_heads=4):
        super(Attention_org, self).__init__()
        self.vis = vis
        self.KV_size = KV_size
        self.channel_num = channel_num
        self.num_attention_heads = num_heads

        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.query3 = nn.ModuleList()
        self.query4 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(num_heads):
            query1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
            query2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
            query3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
            query4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
            key = nn.Linear( self.KV_size,  self.KV_size, bias=False)
            value = nn.Linear(self.KV_size,  self.KV_size, bias=False)
            #把所有的值都重新复制一遍，deepcopy为深复制，完全脱离原来的值，即将被复制对象完全再复制一遍作为独立的新个体单独存在
            self.query1.append(copy.deepcopy(query1))
            self.query2.append(copy.deepcopy(query2))
            self.query3.append(copy.deepcopy(query3))
            self.query4.append(copy.deepcopy(query4))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.out3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
        self.out4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
        self.attn_dropout = Dropout(0.1)
        self.proj_dropout = Dropout(0.1)



    def forward(self, emb1,emb2,emb3,emb4, emb_all):
        multi_head_Q1_list = []
        multi_head_Q2_list = []
        multi_head_Q3_list = []
        multi_head_Q4_list = []
        multi_head_K_list = []
        multi_head_V_list = []
        if emb1 is not None:
            for query1 in self.query1:
                Q1 = query1(emb1)
                multi_head_Q1_list.append(Q1)
        if emb2 is not None:
            for query2 in self.query2:
                Q2 = query2(emb2)
                multi_head_Q2_list.append(Q2)
        if emb3 is not None:
            for query3 in self.query3:
                Q3 = query3(emb3)
                multi_head_Q3_list.append(Q3)
        if emb4 is not None:
            for query4 in self.query4:
                Q4 = query4(emb4)
                multi_head_Q4_list.append(Q4)
        for key in self.key:
            K = key(emb_all)
            multi_head_K_list.append(K)
        for value in self.value:
            V = value(emb_all)
            multi_head_V_list.append(V)
        # print(len(multi_head_Q4_list))

        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1) if emb1 is not None else None
        multi_head_Q2 = torch.stack(multi_head_Q2_list, dim=1) if emb2 is not None else None
        multi_head_Q3 = torch.stack(multi_head_Q3_list, dim=1) if emb3 is not None else None
        multi_head_Q4 = torch.stack(multi_head_Q4_list, dim=1) if emb4 is not None else None
        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)

        multi_head_Q1 = multi_head_Q1.transpose(-1, -2) if emb1 is not None else None
        multi_head_Q2 = multi_head_Q2.transpose(-1, -2) if emb2 is not None else None
        multi_head_Q3 = multi_head_Q3.transpose(-1, -2) if emb3 is not None else None
        multi_head_Q4 = multi_head_Q4.transpose(-1, -2) if emb4 is not None else None

        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_K) if emb1 is not None else None
        attention_scores2 = torch.matmul(multi_head_Q2, multi_head_K) if emb2 is not None else None
        attention_scores3 = torch.matmul(multi_head_Q3, multi_head_K) if emb3 is not None else None
        attention_scores4 = torch.matmul(multi_head_Q4, multi_head_K) if emb4 is not None else None

        attention_scores1 = attention_scores1 / math.sqrt(self.KV_size) if emb1 is not None else None
        attention_scores2 = attention_scores2 / math.sqrt(self.KV_size) if emb2 is not None else None
        attention_scores3 = attention_scores3 / math.sqrt(self.KV_size) if emb3 is not None else None
        attention_scores4 = attention_scores4 / math.sqrt(self.KV_size) if emb4 is not None else None

        attention_probs1 = self.softmax(self.psi(attention_scores1)) if emb1 is not None else None
        attention_probs2 = self.softmax(self.psi(attention_scores2)) if emb2 is not None else None
        attention_probs3 = self.softmax(self.psi(attention_scores3)) if emb3 is not None else None
        attention_probs4 = self.softmax(self.psi(attention_scores4)) if emb4 is not None else None
        # print(attention_probs4.size())

        if self.vis:
            weights =  []
            weights.append(attention_probs1.mean(1))
            weights.append(attention_probs2.mean(1))
            weights.append(attention_probs3.mean(1))
            weights.append(attention_probs4.mean(1))
        else: weights=None

        attention_probs1 = self.attn_dropout(attention_probs1) if emb1 is not None else None
        attention_probs2 = self.attn_dropout(attention_probs2) if emb2 is not None else None
        attention_probs3 = self.attn_dropout(attention_probs3) if emb3 is not None else None
        attention_probs4 = self.attn_dropout(attention_probs4) if emb4 is not None else None

        multi_head_V = multi_head_V.transpose(-1, -2)
        context_layer1 = torch.matmul(attention_probs1, multi_head_V) if emb1 is not None else None
        context_layer2 = torch.matmul(attention_probs2, multi_head_V) if emb2 is not None else None
        context_layer3 = torch.matmul(attention_probs3, multi_head_V) if emb3 is not None else None
        context_layer4 = torch.matmul(attention_probs4, multi_head_V) if emb4 is not None else None

        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous() if emb1 is not None else None
        context_layer2 = context_layer2.permute(0, 3, 2, 1).contiguous() if emb2 is not None else None
        context_layer3 = context_layer3.permute(0, 3, 2, 1).contiguous() if emb3 is not None else None
        context_layer4 = context_layer4.permute(0, 3, 2, 1).contiguous() if emb4 is not None else None
        context_layer1 = context_layer1.mean(dim=3) if emb1 is not None else None
        context_layer2 = context_layer2.mean(dim=3) if emb2 is not None else None
        context_layer3 = context_layer3.mean(dim=3) if emb3 is not None else None
        context_layer4 = context_layer4.mean(dim=3) if emb4 is not None else None

        O1 = self.out1(context_layer1) if emb1 is not None else None
        O2 = self.out2(context_layer2) if emb2 is not None else None
        O3 = self.out3(context_layer3) if emb3 is not None else None
        O4 = self.out4(context_layer4) if emb4 is not None else None
        O1 = self.proj_dropout(O1) if emb1 is not None else None
        O2 = self.proj_dropout(O2) if emb2 is not None else None
        O3 = self.proj_dropout(O3) if emb3 is not None else None
        O4 = self.proj_dropout(O4) if emb4 is not None else None
        return O1,O2,O3,O4, weights




class Mlp(nn.Module):
    def __init__(self, in_channel, mlp_channel):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.0)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block_ViT(nn.Module):
    def __init__(self, vis, channel_num, expand_ratio=4,KV_size=480):
        super(Block_ViT, self).__init__()
        expand_ratio = 4
        self.attn_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.attn_norm3 = LayerNorm(channel_num[2],eps=1e-6)
        self.attn_norm4 = LayerNorm(channel_num[3],eps=1e-6)
        self.attn_norm =  LayerNorm(KV_size,eps=1e-6)
        self.channel_attn = Attention_org(vis, channel_num)

        self.ffn_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.ffn_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.ffn_norm3 = LayerNorm(channel_num[2],eps=1e-6)
        self.ffn_norm4 = LayerNorm(channel_num[3],eps=1e-6)
        self.ffn1 = Mlp(channel_num[0],channel_num[0]*expand_ratio)
        self.ffn2 = Mlp(channel_num[1],channel_num[1]*expand_ratio)
        self.ffn3 = Mlp(channel_num[2],channel_num[2]*expand_ratio)
        self.ffn4 = Mlp(channel_num[3],channel_num[3]*expand_ratio)


    def forward(self, emb1,emb2,emb3,emb4):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3
        org4 = emb4
        for i in range(4):
            var_name = "emb"+str(i+1)  #emb1,emb2,emb3,emb4
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        emb_all = torch.cat(embcat,dim=2)
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)
        cx1,cx2,cx3,cx4, weights = self.channel_attn(cx1,cx2,cx3,cx4,emb_all)
        #残差
        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None
        cx4 = org4 + cx4 if emb4 is not None else None

        org1 = cx1
        org2 = cx2
        org3 = cx3
        org4 = cx4
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None
        x4 = self.ffn_norm4(cx4) if emb4 is not None else None
        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x3 = self.ffn3(x3) if emb3 is not None else None
        x4 = self.ffn4(x4) if emb4 is not None else None
        #残差
        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None
        x4 = x4 + org4 if emb4 is not None else None

        return x1, x2, x3, x4, weights


class Encoder(nn.Module):
    def __init__(self, vis, channel_num, num_layers=4):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.encoder_norm3 = LayerNorm(channel_num[2],eps=1e-6)
        self.encoder_norm4 = LayerNorm(channel_num[3],eps=1e-6)
        for _ in range(num_layers):
            layer = Block_ViT(vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1,emb2,emb3,emb4):
        attn_weights = []
        for layer_block in self.layer:
            emb1,emb2,emb3,emb4, weights = layer_block(emb1,emb2,emb3,emb4)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        return emb1,emb2,emb3,emb4, attn_weights


class ChannelTransformer(nn.Module):
    def __init__(self,  vis=False, img_size=256, channel_num=[64, 128, 256, 512], patchSize=[32, 16, 8, 4]):
        super().__init__()

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]
        self.embeddings_1 = Channel_Embeddings(self.patchSize_1, img_size=img_size,    in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(self.patchSize_2, img_size=img_size//2, in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(self.patchSize_3, img_size=img_size//4, in_channels=channel_num[2])
        self.embeddings_4 = Channel_Embeddings(self.patchSize_4, img_size=img_size//8, in_channels=channel_num[3])
        self.encoder = Encoder( vis, channel_num)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,scale_factor=(self.patchSize_1,self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,scale_factor=(self.patchSize_2,self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1,scale_factor=(self.patchSize_3,self.patchSize_3))
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1,scale_factor=(self.patchSize_4,self.patchSize_4))

    def forward(self,en1,en2,en3,en4):

        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        encoded1, encoded2, encoded3, encoded4, attn_weights = self.encoder(emb1,emb2,emb3,emb4)  # (B, n_patch, hidden)
        x1 = self.reconstruct_1(encoded1) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None
        x3 = self.reconstruct_3(encoded3) if en3 is not None else None
        x4 = self.reconstruct_4(encoded4) if en4 is not None else None

        x1 = x1 + en1  if en1 is not None else None
        x2 = x2 + en2  if en2 is not None else None
        x3 = x3 + en3  if en3 is not None else None
        x4 = x4 + en4  if en4 is not None else None

        return x1, x2, x3, x4, attn_weights
    
#实现了位置编码
class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=512):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, 256, 512)) #8x

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings
    
class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=False):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input): 
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs
    
#实现了自注意力机制，相当于unet的bottleneck层
class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,  #512
        depth,  #4
        heads,  #8
        mlp_dim,  #4096
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers)


    def forward(self, x):
        return self.net(x)
    
#PixelwiseNorm代替了BatchNorm
class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y



class MinibatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super().__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y





# ==========================================================
# Equalized learning rate blocks:
# extending Conv2D and Deconv2D layers for equalized learning rate logic
# ==========================================================
class _equalized_conv2d(nn.Module):
    """ conv2d with the concept of equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out:  output channels
            :param k_size: kernel size (h, w) should be a tuple or a single integer
            :param stride: stride for conv
            :param pad: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """

        super().__init__()

        # define the weight and bias if to be used
        self.weight = nn.Parameter(nn.init.normal_(
            torch.empty(c_out, c_in, *_pair(k_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = np.prod(_pair(k_size)) * c_in  # value of fan_in
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the network
        :param x: input
        :return: y => output
        """
        

        return conv2d(input=x,
                      weight=self.weight * self.scale,  # scale the weight on runtime
                      bias=self.bias if self.use_bias else None,
                      stride=self.stride,
                      padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


class _equalized_deconv2d(nn.Module):
    """ Transpose convolution using the equalized learning rate
        Args:
            :param c_in: input channels
            :param c_out: output channels
            :param k_size: kernel size
            :param stride: stride for convolution transpose
            :param pad: padding
            :param bias: whether to use bias or not
    """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, bias=True):
        """ constructor for the class """

        super().__init__()

        # define the weight and bias if to be used
        self.weight = nn.Parameter(nn.init.normal_(
            torch.empty(c_in, c_out, *_pair(k_size))
        ))

        self.use_bias = bias
        self.stride = stride
        self.pad = pad

        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out).fill_(0))

        fan_in = c_in  # value of fan_in for deconv
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """

        return conv_transpose2d(input=x,
                                weight=self.weight * self.scale,  # scale the weight on runtime
                                bias=self.bias if self.use_bias else None,
                                stride=self.stride,
                                padding=self.pad)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))



#basic block of the encoding part of the genarater
#编码器的基本卷积块
class conv_block(nn.Module):
    """
    Convolution Block 
    with two convolution layers
    """
    def __init__(self, in_ch, out_ch,use_eql=True):
        super(conv_block, self).__init__()
        
        if use_eql:
            self.conv_1=  _equalized_conv2d(in_ch, out_ch, (1, 1),
                                            pad=0, bias=True)
            self.conv_2 = _equalized_conv2d(out_ch, out_ch, (3, 3),
                                            pad=1, bias=True)
            self.conv_3 = _equalized_conv2d(out_ch, out_ch, (3, 3),
                                            pad=1, bias=True)

        else:
            self.conv_1 = Conv2d(in_ch, out_ch, (3, 3),
                                 padding=1, bias=True)
            self.conv_2 = Conv2d(out_ch, out_ch, (3, 3),
                                 padding=1, bias=True)

        # pixel_wise feature normalizer:
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """

        #y = interpolate(x, scale_factor=2)
        y=self.conv_1(self.lrelu(self.pixNorm(x)))
        residual=y
        y=self.conv_2(self.lrelu(self.pixNorm(y)))
        y=self.conv_3(self.lrelu(self.pixNorm(y)))
        y=y+residual


        return y




#basic up convolution block of the encoding part of the genarater
#编码器的基本卷积块
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch,use_eql=True):
        super(up_conv, self).__init__()
        if use_eql:
            self.conv_1=  _equalized_conv2d(in_ch, out_ch, (1, 1),
                                            pad=0, bias=True)
            self.conv_2 = _equalized_conv2d(out_ch, out_ch, (3, 3),
                                            pad=1, bias=True)
            self.conv_3 = _equalized_conv2d(out_ch, out_ch, (3, 3),
                                            pad=1, bias=True)

        else:
            self.conv_1 = Conv2d(in_ch, out_ch, (3, 3),
                                 padding=1, bias=True)
            self.conv_2 = Conv2d(out_ch, out_ch, (3, 3),
                                 padding=1, bias=True)

        # pixel_wise feature normalizer:
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """

        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
        y=self.conv_1(self.lrelu(self.pixNorm(x)))
        residual=y
        y=self.conv_2(self.lrelu(self.pixNorm(y)))
        y=self.conv_3(self.lrelu(self.pixNorm(y)))        
        y=y+residual

        return y




#判别器的最后一层
class DisFinalBlock(nn.Module):
    """ Final block for the Discriminator """

    def __init__(self, in_channels, use_eql=True):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param use_eql: whether to use equalized learning rate
        """

        super().__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels + 1, in_channels, (3, 3),
                                            pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (4, 4),stride=2,pad=1,
                                            bias=True)

            # final layer emulates the fully connected layer
            self.conv_3 = _equalized_conv2d(in_channels, 1, (1, 1), bias=True)

        else:
            # modules required:
            self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (4, 4), bias=True)

            # final conv layer emulates a fully connected layer
            self.conv_3 = Conv2d(in_channels, 1, (1, 1), bias=True)

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the FinalBlock
        :param x: input
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)

        # define the computations
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # fully connected layer
        y = self.conv_3(y)  # This layer has linear activation

        # flatten the output raw discriminator scores
        return y



#判别器基本卷积块
class DisGeneralConvBlock(nn.Module):
    """ General block in the discriminator  """

    def __init__(self, in_channels, out_channels, use_eql=True):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param use_eql: whether to use equalized learning rate
        """

        super().__init__()

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, in_channels, (3, 3),
                                            pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, out_channels, (3, 3),
                                            pad=1, bias=True)
        else:
            # convolutional modules
            self.conv_1 = Conv2d(in_channels, in_channels, (3, 3),
                                 padding=1, bias=True)
            self.conv_2 = Conv2d(in_channels, out_channels, (3, 3),
                                 padding=1, bias=True)

        self.downSampler = nn.AvgPool2d(2)  # downsampler

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the module
        :param x: input
        :return: y => output
        """
        # define the computations
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)

        return y



        

class from_rgb(nn.Module):
    """
    The RGB image is transformed into a multi-channel feature map to be concatenated with 
    the feature map with the same number of channels in the network
    把RGB图转换为多通道特征图，以便与网络中相同通道数的特征图拼接
    """
    def __init__(self, outchannels, use_eql=True):
        super(from_rgb, self).__init__()
        if use_eql:
            self.conv_1 = _equalized_conv2d(3, outchannels, (1, 1), bias=True)
        else:
            self.conv_1 = nn.Conv2d(3, outchannels, (1, 1),bias=True)
        # pixel_wise feature normalizer:
        self.pixNorm = PixelwiseNorm()

        # leaky_relu:
        self.lrelu = nn.LeakyReLU(0.2)


    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        y = self.pixNorm(self.lrelu(self.conv_1(x)))
        return y

class to_rgb(nn.Module):
    """
    把多通道特征图转换为RGB三通道图，以便输入判别器
    The multi-channel feature map is converted into RGB image for input to the discriminator
    """
    def __init__(self, inchannels, use_eql=True):
        super(to_rgb, self).__init__()
        if use_eql:
            self.conv_1 = _equalized_conv2d(inchannels, 3, (1, 1), bias=True)
        else:
            self.conv_1 = nn.Conv2d(inchannels, 3, (1, 1),bias=True)





    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """

        y = self.conv_1(x)

        return y

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)



class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out




##权重初始化
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)






class U_Transformer(nn.Module):
	"""
	MSG-Unet-GAN的生成器部分
	"""
	def __init__(self,
		img_dim=256,
		patch_dim=16,
		embedding_dim=512,
		num_channels=3,
		num_heads=8,
		num_layers=4,
		hidden_dim=256,
		dropout_rate=0.0,
		attn_dropout_rate=0.0,
		in_ch=3, 
		out_ch=3,
		conv_patch_representation=True,
		positional_encoding_type="learned",
		use_eql=True):
		super(U_Transformer, self).__init__()
		assert embedding_dim % num_heads == 0
		assert img_dim % patch_dim == 0

		self.out_ch=out_ch #输出通道数
		self.in_ch=in_ch #输入通道数
		self.img_dim = img_dim   #输入图片尺寸
		self.embedding_dim = embedding_dim  #512
		self.num_heads = num_heads  #多头注意力中头的数量
		self.patch_dim = patch_dim  #每个patch的尺寸
		self.num_channels = num_channels  #图片通道数?
		self.dropout_rate = dropout_rate  #drop-out比率
		self.attn_dropout_rate = attn_dropout_rate  #注意力模块的dropout比率
		self.conv_patch_representation = conv_patch_representation  #True

		self.num_patches = int((img_dim // patch_dim) ** 2)  #将三通道图片分成多少块
		self.seq_length = self.num_patches  #每个sequence的长度为patches的大小
		self.flatten_dim = 128 * num_channels  #128*3=384

        #线性编码
		self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
		#位置编码
		if positional_encoding_type == "learned":
			self.position_encoding = LearnedPositionalEncoding(
				self.seq_length, self.embedding_dim, self.seq_length
			)
		elif positional_encoding_type == "fixed":
			self.position_encoding = FixedPositionalEncoding(
				self.embedding_dim,
			)

		self.pe_dropout = nn.Dropout(p=self.dropout_rate)

		self.transformer = TransformerModel(
			embedding_dim, #512
			num_layers, #4
			num_heads,  #8
			hidden_dim,  #4096

			self.dropout_rate,
			self.attn_dropout_rate,
        )

		#layer Norm
		self.pre_head_ln = nn.LayerNorm(embedding_dim)

		if self.conv_patch_representation:

			self.Conv_x = nn.Conv2d(
				256,
				self.embedding_dim,  #512
				kernel_size=3,
				stride=1,
				padding=1
		    )

		self.bn = nn.BatchNorm2d(256)
		self.relu = nn.ReLU(inplace=True)



		#modulelist
		self.rgb_to_feature=ModuleList([from_rgb(32),from_rgb(64),from_rgb(128)])
		self.feature_to_rgb=ModuleList([to_rgb(32),to_rgb(64),to_rgb(128),to_rgb(256)])

		self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.Conv1=conv_block(self.in_ch, 16)
		self.Conv1_1 = conv_block(16, 32)
		self.Conv2 = conv_block(32, 32)
		self.Conv2_1 = conv_block(32, 64)
		self.Conv3 = conv_block(64,64)
		self.Conv3_1 = conv_block(64,128)
		self.Conv4 = conv_block(128,128)
		self.Conv4_1 = conv_block(128,256)

		self.Conv5 = conv_block(512,256)

		#self.Conv_x = conv_block(256,512)
		self.mtc = ChannelTransformer(channel_num=[32,64,128,256],
									patchSize=[32, 16, 8, 4])
								

		self.Up5 = up_conv(256, 256)
		self.coatt5 = CCA(F_g=256, F_x=256)
		self.Up_conv5 = conv_block(512, 256)
		self.Up_conv5_1 = conv_block(256, 256)

		self.Up4 = up_conv(256, 128)
		self.coatt4 = CCA(F_g=128, F_x=128)
		self.Up_conv4 = conv_block(256, 128)
		self.Up_conv4_1 = conv_block(128, 128)

		self.Up3 = up_conv(128, 64)
		self.coatt3 = CCA(F_g=64, F_x=64)
		self.Up_conv3 = conv_block(128, 64)
		self.Up_conv3_1 = conv_block(64, 64)

		self.Up2 = up_conv(64, 32)
		self.coatt2 = CCA(F_g=32, F_x=32)
		self.Up_conv2 = conv_block(64, 32)
		self.Up_conv2_1 = conv_block(32, 32)

		self.Conv = nn.Conv2d(32, self.out_ch, kernel_size=1, stride=1, padding=0)

		# self.active = torch.nn.Sigmoid()
		# 
	def reshape_output(self,x): #将transformer的输出resize为原来的特征图尺寸
		x = x.view(
			x.size(0),
			int(self.img_dim / self.patch_dim),
			int(self.img_dim / self.patch_dim),
			self.embedding_dim,
			)#B,16,16,512
		x = x.permute(0, 3, 1, 2).contiguous()

		return x

	def forward(self, x):
		#print(x.shape)

		x_1=self.Maxpool(x)
		x_2=self.Maxpool(x_1)
		x_3=self.Maxpool(x_2)


		e1 = self.Conv1(x)
		#print(e1.shape)
		e1 = self.Conv1_1(e1)
		e2 = self.Maxpool1(e1)
		#32*128*128

		x_1=self.rgb_to_feature[0](x_1)
		#e2=torch.cat((x_1,e2), dim=1)
		e2=x_1+e2
		e2 = self.Conv2(e2)
		e2 = self.Conv2_1(e2)
		e3 = self.Maxpool2(e2)
		#64*64*64

		x_2=self.rgb_to_feature[1](x_2)
		#e3=torch.cat((x_2,e3), dim=1)
		e3=x_2+e3
		e3 = self.Conv3(e3)
		e3 = self.Conv3_1(e3)
		e4 = self.Maxpool3(e3)
		#128*32*32

		x_3=self.rgb_to_feature[2](x_3)
		#e4=torch.cat((x_3,e4), dim=1)
		e4=x_3+e4
		e4 = self.Conv4(e4)
		e4 = self.Conv4_1(e4)
		e5 = self.Maxpool4(e4)
		#256*16*16

		#channel-wise transformer-based attention
		e1,e2,e3,e4,_ = self.mtc(e1,e2,e3,e4)




		#spatial-wise transformer-based attention
		residual=e5
		#中间的隐变量
		#conv_x应该接受256通道，输出512通道的中间隐变量
		e5= self.bn(e5)
		e5=self.relu(e5)
		e5= self.Conv_x(e5) #out->512*16*16 shape->B,512,16,16
		e5= e5.permute(0, 2, 3, 1).contiguous()  # B,512,16,16->B,16,16,512
		e5= e5.view(e5.size(0), -1, self.embedding_dim) #B,16,16,512->B,16*16,512 线性映射层
		e5= self.position_encoding(e5) #位置编码
		e5= self.pe_dropout(e5)	 #预dropout层
		# apply transformer
		e5= self.transformer(e5)
		e5= self.pre_head_ln(e5)	
		e5= self.reshape_output(e5)#out->512*16*16 shape->B,512,16,16
		e5=self.Conv5(e5) #out->256,16,16 shape->B,256,16,16
		#residual是否要加bn和relu？
		e5=e5+residual



		d5 = self.Up5(e5)
		e4_att = self.coatt5(g=d5, x=e4)
		d5 = torch.cat((e4_att, d5), dim=1)
		d5 = self.Up_conv5(d5)
		d5 = self.Up_conv5_1(d5)
		#256
		# out3=self.feature_to_rgb[3](d5)
		# output.append(out3)#32*32orH/8,W/8

		d4 = self.Up4(d5)
		e3_att = self.coatt4(g=d4, x=e3)
		d4 = torch.cat((e3_att, d4), dim=1)
		d4 = self.Up_conv4(d4)
		d4 = self.Up_conv4_1(d4)
		#128
		# out2=self.feature_to_rgb[2](d4)
		# output.append(out2)#64*64orH/4,W/4

		d3 = self.Up3(d4)
		e2_att = self.coatt3(g=d3, x=e2)
		d3 = torch.cat((e2_att, d3), dim=1)
		d3 = self.Up_conv3(d3)
		d3 = self.Up_conv3_1(d3)
		#64
		# out1=self.feature_to_rgb[1](d3)
		# output.append(out1)#128#128orH/2,W/2

		d2 = self.Up2(d3)
		e1_att = self.coatt2(g=d2, x=e1)
		d2 = torch.cat((e1_att, d2), dim=1)
		d2 = self.Up_conv2(d2)
		d2 = self.Up_conv2_1(d2)
		#32
		out0=self.feature_to_rgb[0](d2)

		#out = self.Conv(d2)

		#d1 = self.active(out)
		#output=np.array(output)
		
		return out0

if __name__ == '__main__':
    t = torch.randn(1, 3, 256, 256).cuda()
    model = U_Transformer().cuda()
    res = model(t)
    print(res.shape)
    