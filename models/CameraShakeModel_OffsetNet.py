# from torch.nn import init
import functools
import time

# from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

##############################################################################
# Classes
##############################################################################


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x

class ResnetBlock_woNorm(nn.Module):

    def __init__(self, dim, use_bias):
        super(ResnetBlock_woNorm, self).__init__()

        padAndConv_1 = [
                nn.ReplicationPad2d(2),
                nn.Conv2d(dim, dim, kernel_size=5, bias=use_bias)]

        padAndConv_2 = [
            nn.ReplicationPad2d(2),
            nn.Conv2d(dim, dim, kernel_size=5, bias=use_bias)]

        blocks = padAndConv_1 + [
            nn.ReLU(True)
        ]  + padAndConv_2 
        self.conv_block = nn.Sequential(*blocks)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

def TriResblock(input_nc, use_bias=True):
    Res1 =  ResnetBlock_woNorm(input_nc,  use_bias=use_bias)
    Res2 =  ResnetBlock_woNorm(input_nc,  use_bias=use_bias)
    Res3 =  ResnetBlock_woNorm(input_nc,  use_bias=use_bias)
    return nn.Sequential(Res1,Res2,Res3)

def conv_TriResblock(input_nc,out_nc,stride, use_bias=True):
    Relu = nn.ReLU(True)
    if stride==1:
        pad = nn.ReflectionPad2d(2)
        conv = nn.Conv2d(input_nc,out_nc,kernel_size=5,stride=1,padding=0,bias=use_bias)
    elif stride==2:
        pad = nn.ReflectionPad2d((1,2,1,2))
        conv = nn.Conv2d(input_nc,out_nc,kernel_size=5,stride=2,padding=0,bias=use_bias)
    tri_resblock = TriResblock(out_nc)
    return nn.Sequential(pad,conv,Relu,tri_resblock)

class Bottleneck(nn.Module):
    def __init__(self,nChannels,kernel_size=3):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(nChannels, nChannels*2, kernel_size=1, 
                                padding=0, bias=True)
        self.lReLU1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(nChannels*2, nChannels, kernel_size=kernel_size, 
                                padding=(kernel_size-1)//2, bias=True)
        self.lReLU2 = nn.LeakyReLU(0.2, True)
        self.model = nn.Sequential(self.conv1,self.lReLU1,self.conv2,self.lReLU2)
    def forward(self,x):
        out = self.model(x)
        return out

class OffsetNet_quad(nn.Module):
    # offset for Start and End Points, then calculate a quadratic function
    def __init__(self, input_nc=3, nf=16, n_offset=15, offset_mode='quad', norm_layer=nn.BatchNorm2d,gpu_ids=[]):
        super(OffsetNet_quad,self).__init__()
        self.input_nc = input_nc
        self.nf = nf
        self.n_offset = n_offset   
        self.offset_mode = offset_mode
        if offset_mode == 'quad' or offset_mode == 'bilin':
            output_nc = 2 * 2
        elif offset_mode == 'lin':
            output_nc = 1 * 2
        else:
            output_nc = n_offset*2
        
        use_dropout = False
        use_bias=True

        self.pad_1 = nn.ReflectionPad2d((1,2,1,2))
        self.todepth = SpaceToDepth(block_size=2)
        self.conv_1 = conv_TriResblock(input_nc*4,nf,stride=1,use_bias=True)
        self.conv_2 = conv_TriResblock(nf,nf*2,stride=2,use_bias=True)
        self.conv_3 = conv_TriResblock(nf*2,nf*4,stride=2,use_bias=True)

        self.bottleneck_1 = Bottleneck(nf*4)
        self.uconv_1 = nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, 
                                        bias=use_bias)

        self.bottleneck_2 = Bottleneck(nf*4)        
        self.uconv_2 = nn.ConvTranspose2d(nf*4, nf, kernel_size=4, stride=2, padding=1, 
                                        bias=use_bias)
        self.bottleneck_3 = Bottleneck(nf*2)
        self.uconv_3 = nn.ConvTranspose2d(nf*2, nf*2, kernel_size=4, stride=2, padding=1, 
                                        bias=use_bias)
        self.conv_out_0 = nn.Conv2d(nf*2,output_nc,kernel_size=5,stride=1,padding=2,bias=use_bias)

    def Quad_traj(self,offset10,offset12):
        B,C,H,W = offset10.size()
        N = self.n_offset//2
        t = torch.arange(1,N,step=1,dtype=torch.float32).cuda()
        t = t/N
        t = t.view(-1,1,1,1)
        offset10 = offset10.view(B,1,2,H,W)
        offset12 = offset12.unsqueeze(1)
        offset_12N = 0.5 * ((t + t**2)*offset12 - (t - t**2)*offset10)
        offset_10N = 0.5 * ((t + t**2)*offset10 - (t - t**2)*offset12)
        offset_12N = offset_12N.view(B,-1,H,W)
        offset_10N = offset_10N.view(B,-1,H,W)

        return offset_10N,offset_12N

    def Bilinear_traj(self,offset10,offset12):
        B,C,H,W = offset10.size()
        N = self.n_offset//2
        t = torch.arange(1,N,step=1,dtype=torch.float32).cuda()
        t = t/N
        t = t.view(-1,1,1,1)
        offset10 = offset10.view(B,1,2,H,W)
        offset12 = offset12.unsqueeze(1)
        offset_12N = t * offset12
        offset_10N = t * offset10
        offset_12N = offset_12N.view(B,-1,H,W)
        offset_10N = offset_10N.view(B,-1,H,W)
        return offset_10N,offset_12N
    

    def forward(self,input):     
        scale_0 = input
        B,N,H,W = input.size()
        scale_0_depth = self.todepth(scale_0)
        d_conv1 = self.conv_1(scale_0_depth)
        d_conv2 = self.conv_2(d_conv1)
        d_conv3 = self.conv_3(d_conv2)

        d_conv3 = self.bottleneck_1(d_conv3)
        u_conv1 = self.uconv_1(d_conv3)
        u_conv1 = F.leaky_relu(u_conv1,0.2,True) 
        u_conv1 = torch.cat((u_conv1 , d_conv2),dim=1)
        
        u_conv1 = self.bottleneck_2(u_conv1)
        u_conv2 = self.uconv_2(u_conv1)
        u_conv2 = F.leaky_relu(u_conv2,0.2,True)
        u_conv2 = torch.cat((u_conv2 , d_conv1),dim=1)

        u_conv2 = self.bottleneck_3(u_conv2)
        u_conv3 = self.uconv_3(u_conv2)

        out = self.conv_out_0(F.relu(u_conv3))
        
        # quadratic or bilinear
        if self.offset_mode == 'quad' or self.offset_mode == 'bilin':
            offset_SPoint = out[:,:2,:,:]
            offset_EPoint = out[:,2:,:,:]
            if self.offset_mode == 'quad':
                offset_S_0, offset_0_E = self.Quad_traj(offset_SPoint,offset_EPoint)
            else:
                offset_S_0, offset_0_E = self.Bilinear_traj(offset_SPoint,offset_EPoint)
            
            zeros = torch.zeros(B,2,H,W).cuda()
            out = torch.cat((offset_SPoint,offset_S_0,zeros,offset_0_E,offset_EPoint),dim=1)
        elif self.offset_mode == 'lin':
            # linear
            offset_SPoint = out
            offset_EPoint = 0 - out
            offset_S_0, offset_0_E = self.Bilinear_traj(offset_SPoint,offset_EPoint)
        
            zeros = torch.zeros(B,2,H,W).cuda()
            out = torch.cat((offset_SPoint,offset_S_0,zeros,offset_0_E,offset_EPoint),dim=1)
        return out

class BlurNet(nn.Module):
    def __init__(self):
        super(BlurNet,self).__init__()  
        
        self.Dcn = DeformConv2d(in_channels=1, out_channels=1, kernel_size=1,
						stride=1, padding=0, groups=1)
        
    def forward(self,real_B,offset):

        o1, o2 = torch.chunk(offset, 2, dim=1)  ## These two operation may be neccessary for accurate BP in torch 1.10
        offset = torch.cat((o1, o2), dim=1)     ## suggest not to delete
        
        B,C,H,W = offset.size()
        mask = torch.ones(B,1,H,W).cuda() 
        
        # real_B = self.pad(real_B)
        fake_A = self.Dcn(real_B,offset,mask)
        return fake_A
