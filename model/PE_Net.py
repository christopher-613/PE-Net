import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from functools import partial
from .TAGT_utils import TAGT_block
import torch.nn.functional as F
from torchsummary import summary
import os
nonlinearity = partial(F.relu, inplace=True)



def downsample():
    return nn.MaxPool3d(kernel_size=2, stride=2)
def deconv(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# depth axes 
class DepthWiseConv(nn.Module):
    def __init__ ( self,in_channel,out_channel,kernel_size = 3, stride = 1,padding = 1):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv3d(in_channels=in_channel,
                  out_channels=in_channel,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  groups=in_channel)
        self.point_conv = nn.Conv3d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    
    def forward(self,input):
        out = input.clone()
        out = self.depth_conv(out)
        out = self.point_conv(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#! 1*1 ->1*1*1


class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""


    

#! TAGT_ATTENTION 
class TAGT_Attention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False, length = False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(TAGT_Attention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width
        self.length = length
        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Priority on encoding

        ## Initial values 

        self.f_qr = nn.Parameter(torch.tensor(0.1),  requires_grad=False) 
        self.f_kr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0),  requires_grad=False)


        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()
        # self.print_para()

    def forward(self, x):   #! add length N, C, L , H, W
        if self.length:
            x = x.permute(0, 3, 4, 1, 2) # N, H, W, C, L
        else:
          if self.width:
            x = x.permute(0, 2, 3, 1, 4)  # N, L, H, C, W
          else:
            x = x.permute(0, 2, 4, 1, 3)  # N, L, W, C, H
        N, L, W, C, H = x.shape
        
        x = x.contiguous().view(N * W * L, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)


        # multiply by factors
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.length:  #! ADD
            output = output.permute(0, 3, 4, 1, 2)
        else:
          if self.width:
            output = output.permute(0, 3, 1, 2, 4)  # N, W, C, H
          else:
            output = output.permute(0, 3, 1, 4, 2)
        
        if self.stride > 1:
            output = self.pooling(output)

        return output
    

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))



# TAGT block
class TAGT_block(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=128, dilation=1, norm_layer=None, kernel_size=56, length=16):
        super(TAGT_block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(4,width)
        self.hight_block = TAGT_Attention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = TAGT_Attention(width, width, groups=groups, kernel_size=kernel_size, width=True)
        self.length_block = TAGT_Attention(width, width, groups=groups, kernel_size = length, stride=stride, length=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(4,planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.length_block(out)
        
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out






class ConvDecoder(nn.Module):
    def __init__(self, in_channels):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=False)
    

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out
    
class FFB(nn.Module):
    def __init__(self, features, M=2, r=4, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(FFB, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x1, x2):
        
        feas = torch.cat((x1.unsqueeze_(dim=1), x2.unsqueeze_(dim=1)), dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1).mean((-1))
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v
        


class FFB_Decoder(nn.Module):
    def __init__(self, out_channels):
        super(FFB_Decoder, self).__init__()
        self.conv1 = FFB(out_channels)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.DoubleDecoder = ConvDecoder(out_channels)
        # self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        # residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x1, x2)))
        out = self.DoubleDecoder(out)

        # out = self.relu(self.bn2(self.conv2(out)))
        # out += residual
        # out = self.relu(out)
        return out

    
class DoubleconvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleconvEncoder, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out    


class FFM(nn.Module):
    def __init__(self, features, M=2, r=4, L=32):
        super(FFM, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        feas = torch.cat((x1.unsqueeze_(dim=1), x2.unsqueeze_(dim=1)), dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1).mean((-1))
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

class EFC(nn.Module):
    def __init__(self, dim):  # stat from 64  dim_s-> now   dim-> next layer
        super(EFC, self).__init__()
        self.convn_1 = nn.Conv3d(dim, 1, kernel_size=1)
        self.convTrans = nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2)
        self.conv11 = nn.Conv3d(1, 1, kernel_size=1)
        self.dim_s = int(dim//2)
        self.convn_2 = nn.Conv3d(self.dim_s, 1, kernel_size=1)

    def forward(self, x1, x2):  #enc1 enc2

        conv1_1 = self.convn_1(x2)   
        convTrans = self.convTrans(conv1_1)
        path1 = -1 * (torch.sigmoid(convTrans)) + 1
        path1 = path1.expand(-1, self.dim_s, -1, -1, -1).mul(x1)
        path1 = path1 + x1

        conv1_2 = self.convn_2(x1)   
        path2 = self.conv11(conv1_2)
        path2 =  torch.sigmoid(path2)
        path2 = path2.mul(x1)
        path2 = path1 + x1

        fea_e = path1+path2
        return fea_e


class PE_Net(nn.Module):

    def __init__(self, block, layers, classes=2, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.25, img_size=128, img_depth = 128,imgchan=1):
        super(PE_Net, self).__init__()
       '''
       delete
       
       '''
    def forward(self, x):
        conv1 = self.conv1(x)
        tagt1=self.TAGT1(x)
        enc1 = FFM(conv1, tagt1)
        down1 = self.down(enc1)

        conv2 = self.conv2(down1)
        tagt2=self.TAGT1(down1)
        enc2 = FFM(conv2, tagt2)
        down2 = self.down(enc2)

        f_aug1 = self.edge1(enc1, enc2)

        conv3 = self.conv2(down2)
        tagt3=self.TAGT1(down2)
        enc3 = FFM(conv3, tagt3)
        down3 = self.down(enc3)

        f_aug2 = self.edge1(enc2, enc3)

        conv_bri = self.conv_bri(down3)
        tagt_bri = self.TAGT_bri(down3)
        bridge = FFM(conv_bri, tagt_bri)

        f_aug3 = self.edge1(enc3, bridge)

        up3 = self.up3(bridge)
      
        dec3 = self.decoder3(up3, f_aug3)

        up2 = self.up2(dec3)
        # up2 = torch.cat((up2, x2), dim=1)
        dec2 = self.decoder2(up2, f_aug2)

        up1 = self.up1(dec2)
        # up1 = torch.cat((up1, x3), dim=1)
        dec1 = self.decoder1(up1, f_aug1)

        final = self.final(dec1)
        final = F.sigmoid(final)
        return  final

# net = PE_Net(TAGT_block, [1,2,4,2], s= 0.25)
# summary(net,(3,128,128,128))
    
