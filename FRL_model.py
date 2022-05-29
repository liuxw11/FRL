import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math      

def initDCTKernel(N):
    kernel = np.zeros((N, N, N*N))
    cnum = 0
    for i in range(N):
        for j in range(N):
            ivec = np.linspace(0.5 * math.pi / N * i, (N - 0.5) * math.pi / N * i, num=N)
            ivec = np.cos(ivec)
            jvec = np.linspace(0.5 * math.pi / N * j, (N - 0.5) * math.pi / N * j, num=N)
            jvec = np.cos(jvec)
            slice = np.outer(ivec, jvec)

            if i==0 and j==0:
                slice = slice / N
            elif i*j==0:
                slice = slice * np.sqrt(2) / N
            else:
                slice = slice * 2.0 / N

            kernel[:,:,cnum] = slice
            cnum = cnum + 1
    kernel = kernel[np.newaxis, :]
    kernel = np.transpose(kernel, (3,0,1,2))
    return kernel

def initIDCTKernel(N):
    kernel = np.zeros((N, N, N*N))
    for i_ in range(N):
        i = N - i_ - 1
        for j_ in range(N):
            j = N - j_ - 1
            ivec = np.linspace(0, (i+0.5)*math.pi/N * (N-1), num=N)
            ivec = np.cos(ivec)
            jvec = np.linspace(0, (j+0.5)*math.pi/N * (N-1), num=N)
            jvec = np.cos(jvec)
            slice = np.outer(ivec, jvec)

            ic = np.sqrt(2.0 / N) * np.ones(N)
            ic[0] = np.sqrt(1.0 / N)
            jc = np.sqrt(2.0 / N) * np.ones(N)
            jc[0] = np.sqrt(1.0 / N)
            cmatrix = np.outer(ic, jc)

            slice = slice * cmatrix
            slice = slice.reshape((1, N*N))
            slice = slice[np.newaxis, :]
            kernel[i_, j_, :] = slice / (N * N)
    kernel = kernel[np.newaxis, :]
    kernel = np.transpose(kernel, (0,3,1,2))
    return kernel
    
def conv_3x3(in_channel, out_channel, stride=1, bias=False, padding=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, bias=bias)
def conv_1x1(in_channel, out_channel):
        return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
    
class FRL(nn.Module):
    def __init__(self):
        super(FRL, self).__init__()
        self.kernelSize = 7
        self.feat = 32
        self.channelNum = self.kernelSize*self.kernelSize
        self.inKernel = nn.Parameter(torch.Tensor(initDCTKernel(self.kernelSize)))
        self.outKernel = nn.Parameter(torch.Tensor(initIDCTKernel(self.kernelSize)))
        self.inKernel.requires_grad = False
        self.conv_1dt = nn.Sequential(conv_1x1(self.channelNum, self.feat), nn.ReLU(inplace=True), nn.BatchNorm2d(self.feat))
        self.conv_1dg = nn.Sequential(conv_1x1(self.channelNum, self.feat), nn.ReLU(inplace=True), nn.BatchNorm2d(self.feat))
        self.conv_2dt = nn.Sequential(conv_3x3(self.feat, self.feat), nn.ReLU(inplace=True), nn.BatchNorm2d(self.feat))
        self.conv_2dg = nn.Sequential(conv_3x3(self.feat, self.feat), nn.ReLU(inplace=True), nn.BatchNorm2d(self.feat))
        self.conv_3dt = nn.Sequential(conv_3x3(self.feat, self.feat), nn.ReLU(inplace=True), nn.BatchNorm2d(self.feat)) 
        self.conv_3dg = nn.Sequential(conv_3x3(self.feat, self.feat), nn.ReLU(inplace=True), nn.BatchNorm2d(self.feat)) 
        self.conv_4d = nn.Sequential(conv_3x3(2*self.feat+2*self.channelNum, 2*self.feat), nn.ReLU(inplace=True), nn.BatchNorm2d(2*self.feat)) 
        self.conv_5d = nn.Sequential(conv_3x3(2*self.feat, 2*self.feat), nn.ReLU(inplace=True), nn.BatchNorm2d(2*self.feat)) 
        self.conv_6d = nn.Sequential(conv_3x3(2*self.feat, 2*self.feat), nn.ReLU(inplace=True), nn.BatchNorm2d(2*self.feat)) 
        self.conv_7d = nn.Sequential(conv_3x3(2*self.feat, 2*self.feat), nn.ReLU(inplace=True), nn.BatchNorm2d(2*self.feat)) 
        self.conv_8d = nn.Sequential(conv_3x3(2*self.feat, 2*self.feat), nn.ReLU(inplace=True), nn.BatchNorm2d(2*self.feat)) 
        self.conv_9d = nn.Sequential(conv_3x3(2*self.feat, 2*self.feat), nn.ReLU(inplace=True), nn.BatchNorm2d(2*self.feat)) 
        self.conv_10d = nn.Sequential(conv_1x1(2*self.feat, self.channelNum)) 


    def forward(self, target,guidance):
        out_g0 = F.conv2d(input=guidance, weight=self.inKernel, padding=self.kernelSize - 1)
        out_t0 = F.conv2d(input=target, weight=self.inKernel, padding=self.kernelSize - 1)
        xt = self.conv_1dt(out_t0)
        xt = self.conv_2dt(xt)
        xt = self.conv_3dt(xt)
        xg = self.conv_1dg(out_g0)
        xg = self.conv_2dg(xg)
        xg = self.conv_3dg(xg)
        xf =  torch.cat([out_t0,out_g0,xt,xg], dim=1)
        x = self.conv_4d(xf)
        x = self.conv_5d(x)
        x = self.conv_6d(x)
        x = self.conv_7d(x)
        x = self.conv_8d(x)
        x = self.conv_9d(x)
        res = self.conv_10d(x)
        out_f = out_t0 - res
        denoisedTarget = F.conv2d(input=out_f, weight=self.outKernel, padding=0)
        return denoisedTarget

        
