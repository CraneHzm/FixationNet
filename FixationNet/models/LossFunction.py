# Copyright (c) Hu Zhiming 2020/5/1 jimmyhu@pku.edu.cn All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# define the angular loss.
class AngularLoss2(nn.Module):
    def __init__(self, p=2., eps=1e-6, keepdim=False, size_average=True):
        super().__init__()
        self.norm = p
        self.eps = eps
        self.keepdim = keepdim
        self.size_average = size_average
    def forward(self, input, target):
        loss = F.pairwise_distance(input, target, self.norm, self.eps, self.keepdim)
        if self.size_average:
            return loss.mean()
        return loss.sum()


# define the angular loss.
class AngularLoss1(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average
    def forward(self,input,target):
        n = torch.pow(input - target, 2)
        #print(n.shape)
        loss = torch.sqrt(torch.sum(n, 1))
        #print(loss.shape)
        
        if self.size_average:
            return loss.mean()
        return loss.sum()


# define the angular loss.
class AngularLoss(nn.Module):
    def __init__(self, size_average=True, eps=1e-6):
        super().__init__()
        self.size_average = size_average
        self.eps = eps
    def forward(self,input,target):
        loss = CalAngularDist(input, target, self.eps)        
        if self.size_average:
            return loss.mean()
        return loss.sum()

    
# Calculate the angular distance (visual angle) between two inputs
def CalAngularDist(x, y, eps=1e-6):
    #the parameters of our Hmd (HTC Vive).
    #Vertical FOV.
    VerticalFov = math.pi*110/180
    #Size of a half screen.
    ScreenWidth = 1080
    ScreenHeight = 1200
    ScreenCenterX = 0.5*ScreenWidth
    ScreenCenterY = 0.5*ScreenHeight
    #the pixel distance between eye and the screen center.
    ScreenDist = 0.5* ScreenHeight/math.tan(VerticalFov/2)
    
    x1, x2 = x.split(1, 1)
    y1, y2 = y.split(1, 1)
    #transform the angular coords to screen coords.
    # the X coord.
    x1 = ScreenDist * torch.tan(math.pi*x1 / 180) + 0.5*ScreenWidth
    y1 = ScreenDist * torch.tan(math.pi*y1 / 180) + 0.5*ScreenWidth
    # the Y coord.
    x2 = ScreenDist * torch.tan(-math.pi*x2 / 180) + 0.5*ScreenHeight
    y2 = ScreenDist * torch.tan(-math.pi*y2 / 180) + 0.5*ScreenHeight
    
    #the square of the distance between eye and x
    eye2x = ScreenDist*ScreenDist + torch.pow(x1 - ScreenCenterX, 2) + torch.pow(x2 - ScreenCenterY, 2)
    #the square of the distance between eye and y
    eye2y = ScreenDist*ScreenDist + torch.pow(y1 - ScreenCenterX, 2) + torch.pow(y2 - ScreenCenterY, 2)
    #the square of the distance between x and y
    x2y = torch.pow(y1 - x1, 2) + torch.pow(y2 - x2, 2)
    #cos value
    cos_value = (eye2x + eye2y - x2y)/(2*torch.sqrt(eye2x)*torch.sqrt(eye2y))
    dist = acos_safe(cos_value, eps)/math.pi*180
    return dist
    
def acos_safe(x, eps=1e-6):
    slope = np.arccos(1-eps) / eps
    buf = torch.empty_like(x)
    good = abs(x) <= 1-eps
    bad = ~good
    sign = torch.sign(x[bad])
    buf[good] = torch.acos(x[good])
    buf[bad] = torch.acos(sign * (1 - eps)) - slope*sign*(abs(x[bad]) - 1 + eps)
    return buf


# define the huber loss.    
class HuberLoss(nn.Module):
    def __init__(self, beta= 1.0, size_average=True):
        super().__init__()
        self.beta = beta
        self.size_average = size_average
    def forward(self,input,target):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        n = torch.abs(input - target)
        cond = n < self.beta
        loss = torch.where(cond, 0.5 * n ** 2, self.beta*(n - 0.5 * self.beta))
        if self.size_average:
            return loss.mean()
        return loss.sum() 
    
    
# define the custom loss.    
class CustomLoss(nn.Module):
    def __init__(self, beta= 1.0, size_average=True):
        super().__init__()
        self.beta = beta
        self.size_average = size_average
    def forward(self,input,target):
        
        n = torch.abs(input - target)
        cond = n < self.beta
        loss = torch.where(cond, 0*n, n - self.beta)
        if self.size_average:
            return loss.mean()
        return loss.sum()     