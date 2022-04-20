# Copyright (c) Hu Zhiming 2020/6/1 jimmyhu@pku.edu.cn All Rights Reserved.

import torch
import torch.nn as nn
#from torchvision import models
import torch.nn.init as init
import torch.nn.functional as F
from math import floor    
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FixationNet(nn.Module):
    def __init__(self, gazeSeqSize, headSeqSize, objectSeqSize, saliencySize, clusterPath):
        super().__init__()
        # the input params
        self.gazeSeqSize = gazeSeqSize
        self.headSeqSize = headSeqSize
        self.objectSeqSize = objectSeqSize
        self.saliencySize = saliencySize
        
        print('gazeSeqSize: {}'.format(self.gazeSeqSize))
        print('headSeqSize: {}'.format(self.headSeqSize))
        print('taskObjectSeqSize: {}'.format(self.objectSeqSize))
        print('saliencySize: {}'.format(self.saliencySize))
        
        # preset params
        self.gazeFeatureNum = 2
        self.gazeSeqLength = int(self.gazeSeqSize/self.gazeFeatureNum)
        self.headFeatureNum = 2
        self.headSeqLength = int(self.headSeqSize/self.headFeatureNum)
        # the nearest 3 task-related objects, 3*4
        self.objectFeatureNum = 12
        self.objectSeqLength = int(self.objectSeqSize/self.objectFeatureNum)
        self.saliencyWidth = 24
        self.saliencyNum = int(self.saliencySize/(self.saliencyWidth*self.saliencyWidth))
        
        # load the cluster centers
        self.cluster = torch.from_numpy(np.load(clusterPath)).float().to(device)
        self.clusterSize = self.cluster.shape[0]
        print('Cluster Size: {}'.format(self.clusterSize))
        
        
        # GazeSeqCNN1D Module
        gazeSeqCNN1D_outChannels1 = 32
        gazeSeqCNN1D_poolingRate1 = 2
        gazeSeqCNN1D_kernelSize1 = 1
        gazeSeqCNN1D_featureSize1 = floor((self.gazeSeqLength - gazeSeqCNN1D_kernelSize1 + 1)/gazeSeqCNN1D_poolingRate1)
        self.gazeSeqCNN1D_outputSize = gazeSeqCNN1D_featureSize1 * gazeSeqCNN1D_outChannels1
        self.gazeSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.gazeFeatureNum, out_channels=gazeSeqCNN1D_outChannels1,kernel_size=gazeSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(gazeSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.LeakyReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(gazeSeqCNN1D_poolingRate1),
             )
        
        gazeSeqCNN_dropoutRate = 0.0
        self.gazeSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = gazeSeqCNN_dropoutRate)
             )       
        
        # HeadSeqCNN1D Module
        headSeqCNN1D_outChannels1 = 64
        headSeqCNN1D_poolingRate1 = 2
        headSeqCNN1D_kernelSize1 = 1
        headSeqCNN1D_featureSize1 = floor((self.headSeqLength - headSeqCNN1D_kernelSize1 + 1)/headSeqCNN1D_poolingRate1)
        self.headSeqCNN1D_outputSize = headSeqCNN1D_featureSize1 * headSeqCNN1D_outChannels1
        self.headSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.headFeatureNum, out_channels=headSeqCNN1D_outChannels1,kernel_size=headSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(headSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.LeakyReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(headSeqCNN1D_poolingRate1),
            #nn.Conv1d(in_channels=headSeqCNN1D_outChannels1, out_channels=headSeqCNN1D_outChannels2,kernel_size=headSeqCNN1D_kernelSize2),
            #nn.BatchNorm1d(headSeqCNN1D_outChannels2),
            #nn.Sigmoid(),
            #nn.MaxPool1d(headSeqCNN1D_poolingRate2),            
             )
        
        headSeqCNN_dropoutRate = 0.0
        self.headSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = headSeqCNN_dropoutRate)
             )       

        
        # objectSeqCNN1D Module
        objectSeqCNN1D_outChannels1 = 64
        objectSeqCNN1D_poolingRate1 = 2
        objectSeqCNN1D_kernelSize1 = 1
        objectSeqCNN1D_featureSize1 = floor((self.objectSeqLength - objectSeqCNN1D_kernelSize1 + 1)/objectSeqCNN1D_poolingRate1)
        objectSeqCNN1D_outChannels2 = 32
        objectSeqCNN1D_poolingRate2 = 2
        objectSeqCNN1D_kernelSize2 = 1
        objectSeqCNN1D_featureSize2 = floor((objectSeqCNN1D_featureSize1 - objectSeqCNN1D_kernelSize2 + 1)/objectSeqCNN1D_poolingRate2)
        self.objectSeqCNN1D_outputSize = objectSeqCNN1D_featureSize2 * objectSeqCNN1D_outChannels2
        self.objectSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.objectFeatureNum, out_channels=objectSeqCNN1D_outChannels1,kernel_size=objectSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(objectSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.LeakyReLU(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(objectSeqCNN1D_poolingRate1),
            nn.Conv1d(in_channels=objectSeqCNN1D_outChannels1, out_channels=objectSeqCNN1D_outChannels2,kernel_size=objectSeqCNN1D_kernelSize2),
            nn.BatchNorm1d(objectSeqCNN1D_outChannels2),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.MaxPool1d(objectSeqCNN1D_poolingRate2),
             )
        
        objectSeqCNN_dropoutRate = 0.0
        self.objectSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = objectSeqCNN_dropoutRate)
             )       
        
         
        # SalCNN Module
        salCNN_outChannels1 = 8
        salCNN_poolingRate1 = 2
        salCNN_kernelSize1 = 1
        salCNN_padding1 = int((salCNN_kernelSize1-1)/2)
        salCNN_imageSize1 = floor((self.saliencyWidth - salCNN_kernelSize1 + 2*salCNN_padding1 + 1)/salCNN_poolingRate1)
        self.salCNN_outputSize = salCNN_imageSize1 * salCNN_imageSize1 * salCNN_outChannels1        
        self.SalCNN = nn.Sequential(
            nn.Conv2d(self.saliencyNum, salCNN_outChannels1, kernel_size=salCNN_kernelSize1, padding=salCNN_padding1),
            nn.BatchNorm2d(salCNN_outChannels1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool2d(kernel_size=salCNN_poolingRate1),
             )
        
        salCNN_dropoutRate = 0.5
        self.SalCNNDropout = nn.Sequential(
            nn.Dropout(p = salCNN_dropoutRate)
             )       
        
        
        # PrdFC Module
        prdFC_inputSize = self.gazeSeqCNN1D_outputSize + self.headSeqCNN1D_outputSize + self.objectSeqCNN1D_outputSize + self.salCNN_outputSize
        prdFC_linearSize1 = 128
        prdFC_linearSize2 = self.clusterSize
        prdFC_dropoutRate = 0.5
        # the prediction fc layer
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.Softmax(dim = 1),
             )
        
        
    def forward1(self, x):
        currentGaze = x[:, 0:2]
        index = self.gazeSeqSize
        gazeSeq = x[:, 0:index]
        headSeq = x[:, index: index+self.headSeqSize]
        index += self.headSeqSize
        objectSeq = x[:, index: index+self.objectSeqSize]
        index += self.objectSeqSize
        saliencyMap = x[:, index: index+self.saliencySize]
        
        gazeSeq = gazeSeq.reshape(-1, self.gazeSeqLength, self.gazeFeatureNum)
        gazeSeq = gazeSeq.permute(0,2,1)        
        gazeSeqOut = self.gazeSeqCNN1D(gazeSeq)
        gazeSeqOut = gazeSeqOut.reshape(-1, self.gazeSeqCNN1D_outputSize)
        gazeSeqOut = self.gazeSeqCNNDropout(gazeSeqOut)
        
        headSeq = headSeq.reshape(-1, self.headSeqLength, self.headFeatureNum)
        headSeq = headSeq.permute(0,2,1)        
        headSeqOut = self.headSeqCNN1D(headSeq)
        headSeqOut = headSeqOut.reshape(-1, self.headSeqCNN1D_outputSize)
        headSeqOut = self.headSeqCNNDropout(headSeqOut)
        
        objectSeq = objectSeq.reshape(-1, self.objectSeqLength, self.objectFeatureNum)
        objectSeq = objectSeq.permute(0,2,1)        
        objectSeqOut = self.objectSeqCNN1D(objectSeq)
        objectSeqOut = objectSeqOut.reshape(-1, self.objectSeqCNN1D_outputSize)
        objectSeqOut = self.objectSeqCNNDropout(objectSeqOut)

        seqOut = torch.cat((gazeSeqOut, headSeqOut), 1)
        seqOut = torch.cat((seqOut, objectSeqOut), 1)
        
        saliencyMap = saliencyMap.reshape(saliencyMap.size(0), self.saliencyNum, self.saliencyWidth, self.saliencyWidth)
        saliencyFeatures = self.SalCNN(saliencyMap)
        saliencyOut = saliencyFeatures.reshape(saliencyMap.size(0), -1)        
        saliencyOut = self.SalCNNDropout(saliencyOut)
        
        prdInput = torch.cat((seqOut, saliencyOut), 1)
        prdOut = self.PrdFC(prdInput)
        out = prdOut.mm(self.cluster)
        out = out + currentGaze
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out  
    

# FixationNet for forecasting eye fixations in long-term future (over 300 ms)
# CurrentGaze is not used.
class FixationNet300(nn.Module):
    def __init__(self, gazeSeqSize, headSeqSize, objectSeqSize, saliencySize, clusterPath):
        super().__init__()
        # the input params
        self.gazeSeqSize = gazeSeqSize
        self.headSeqSize = headSeqSize
        self.objectSeqSize = objectSeqSize
        self.saliencySize = saliencySize
        
        print('gazeSeqSize: {}'.format(self.gazeSeqSize))
        print('headSeqSize: {}'.format(self.headSeqSize))
        print('taskObjectSeqSize: {}'.format(self.objectSeqSize))
        print('saliencySize: {}'.format(self.saliencySize))
        
        # preset params
        self.gazeFeatureNum = 2
        self.gazeSeqLength = int(self.gazeSeqSize/self.gazeFeatureNum)
        self.headFeatureNum = 2
        self.headSeqLength = int(self.headSeqSize/self.headFeatureNum)
        self.objectFeatureNum = 12
        self.objectSeqLength = int(self.objectSeqSize/self.objectFeatureNum)
        self.saliencyWidth = 24
        self.saliencyNum = int(self.saliencySize/(self.saliencyWidth*self.saliencyWidth))
        
        # load the cluster centers
        self.cluster = torch.from_numpy(np.load(clusterPath)).float().to(device)
        self.clusterSize = self.cluster.shape[0]
        print('Cluster Size: {}'.format(self.clusterSize))
        
        
        # GazeSeqCNN1D Module
        gazeSeqCNN1D_outChannels1 = 32
        gazeSeqCNN1D_poolingRate1 = 2
        gazeSeqCNN1D_kernelSize1 = 1
        gazeSeqCNN1D_featureSize1 = floor((self.gazeSeqLength - gazeSeqCNN1D_kernelSize1 + 1)/gazeSeqCNN1D_poolingRate1)
        self.gazeSeqCNN1D_outputSize = gazeSeqCNN1D_featureSize1 * gazeSeqCNN1D_outChannels1
        self.gazeSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.gazeFeatureNum, out_channels=gazeSeqCNN1D_outChannels1,kernel_size=gazeSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(gazeSeqCNN1D_outChannels1),
            #nn.ReLU(),
            #nn.LeakyReLU(),
            nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(gazeSeqCNN1D_poolingRate1),
             )
        
        gazeSeqCNN_dropoutRate = 0.0
        self.gazeSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = gazeSeqCNN_dropoutRate)
             )       
        
        # HeadSeqCNN1D Module
        headSeqCNN1D_outChannels1 = 64
        headSeqCNN1D_poolingRate1 = 2
        headSeqCNN1D_kernelSize1 = 1
        headSeqCNN1D_featureSize1 = floor((self.headSeqLength - headSeqCNN1D_kernelSize1 + 1)/headSeqCNN1D_poolingRate1)
        self.headSeqCNN1D_outputSize = headSeqCNN1D_featureSize1 * headSeqCNN1D_outChannels1
        self.headSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.headFeatureNum, out_channels=headSeqCNN1D_outChannels1,kernel_size=headSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(headSeqCNN1D_outChannels1),
            #nn.ReLU(),
            #nn.LeakyReLU(),
            nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(headSeqCNN1D_poolingRate1),
            #nn.Conv1d(in_channels=headSeqCNN1D_outChannels1, out_channels=headSeqCNN1D_outChannels2,kernel_size=headSeqCNN1D_kernelSize2),
            #nn.BatchNorm1d(headSeqCNN1D_outChannels2),
            #nn.Sigmoid(),
            #nn.MaxPool1d(headSeqCNN1D_poolingRate2),            
             )
        
        headSeqCNN_dropoutRate = 0.0
        self.headSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = headSeqCNN_dropoutRate)
             )       
        
        
        # objectSeqCNN1D Module
        objectSeqCNN1D_outChannels1 = 64
        objectSeqCNN1D_poolingRate1 = 2
        objectSeqCNN1D_kernelSize1 = 1
        objectSeqCNN1D_featureSize1 = floor((self.objectSeqLength - objectSeqCNN1D_kernelSize1 + 1)/objectSeqCNN1D_poolingRate1)
        objectSeqCNN1D_outChannels2 = 32
        objectSeqCNN1D_poolingRate2 = 2
        objectSeqCNN1D_kernelSize2 = 1
        objectSeqCNN1D_featureSize2 = floor((objectSeqCNN1D_featureSize1 - objectSeqCNN1D_kernelSize2 + 1)/objectSeqCNN1D_poolingRate2)
        self.objectSeqCNN1D_outputSize = objectSeqCNN1D_featureSize2 * objectSeqCNN1D_outChannels2
        self.objectSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.objectFeatureNum, out_channels=objectSeqCNN1D_outChannels1,kernel_size=objectSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(objectSeqCNN1D_outChannels1),
            #nn.ReLU(),
            nn.Sigmoid(),
            #nn.LeakyReLU(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(objectSeqCNN1D_poolingRate1),
            nn.Conv1d(in_channels=objectSeqCNN1D_outChannels1, out_channels=objectSeqCNN1D_outChannels2,kernel_size=objectSeqCNN1D_kernelSize2),
            nn.BatchNorm1d(objectSeqCNN1D_outChannels2),
            #nn.ReLU(),
            nn.Sigmoid(),
            nn.MaxPool1d(objectSeqCNN1D_poolingRate2),
             )
        
        objectSeqCNN_dropoutRate = 0.0
        self.objectSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = objectSeqCNN_dropoutRate)
             )       
        
         
        # SalCNN Module
        salCNN_outChannels1 = 8
        salCNN_poolingRate1 = 2
        salCNN_kernelSize1 = 1
        salCNN_padding1 = int((salCNN_kernelSize1-1)/2)
        salCNN_imageSize1 = floor((self.saliencyWidth - salCNN_kernelSize1 + 2*salCNN_padding1 + 1)/salCNN_poolingRate1)
        self.salCNN_outputSize = salCNN_imageSize1 * salCNN_imageSize1 * salCNN_outChannels1        
        self.SalCNN = nn.Sequential(
            nn.Conv2d(self.saliencyNum, salCNN_outChannels1, kernel_size=salCNN_kernelSize1, padding=salCNN_padding1),
            nn.BatchNorm2d(salCNN_outChannels1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool2d(kernel_size=salCNN_poolingRate1),
             )
        
        salCNN_dropoutRate = 0.5
        self.SalCNNDropout = nn.Sequential(
            nn.Dropout(p = salCNN_dropoutRate)
             )       
        
        
        # PrdFC Module
        prdFC_inputSize = self.gazeSeqCNN1D_outputSize + self.headSeqCNN1D_outputSize + self.objectSeqCNN1D_outputSize + self.salCNN_outputSize
        prdFC_linearSize1 = 128
        prdFC_linearSize2 = self.clusterSize
        prdFC_dropoutRate = 0.5
        # the prediction fc layer
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            #nn.ReLU(),
            nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.Softmax(dim = 1),
             )
        
        
    def forward1(self, x):
        #currentGaze = x[:, 0:2]
        index = self.gazeSeqSize
        gazeSeq = x[:, 0:index]
        headSeq = x[:, index: index+self.headSeqSize]
        index += self.headSeqSize
        objectSeq = x[:, index: index+self.objectSeqSize]
        index += self.objectSeqSize
        saliencyMap = x[:, index: index+self.saliencySize]
        
        gazeSeq = gazeSeq.reshape(-1, self.gazeSeqLength, self.gazeFeatureNum)
        gazeSeq = gazeSeq.permute(0,2,1)        
        gazeSeqOut = self.gazeSeqCNN1D(gazeSeq)
        gazeSeqOut = gazeSeqOut.reshape(-1, self.gazeSeqCNN1D_outputSize)
        gazeSeqOut = self.gazeSeqCNNDropout(gazeSeqOut)
        
        headSeq = headSeq.reshape(-1, self.headSeqLength, self.headFeatureNum)
        headSeq = headSeq.permute(0,2,1)        
        headSeqOut = self.headSeqCNN1D(headSeq)
        headSeqOut = headSeqOut.reshape(-1, self.headSeqCNN1D_outputSize)
        headSeqOut = self.headSeqCNNDropout(headSeqOut)
        
        objectSeq = objectSeq.reshape(-1, self.objectSeqLength, self.objectFeatureNum)
        objectSeq = objectSeq.permute(0,2,1)        
        objectSeqOut = self.objectSeqCNN1D(objectSeq)
        objectSeqOut = objectSeqOut.reshape(-1, self.objectSeqCNN1D_outputSize)
        objectSeqOut = self.objectSeqCNNDropout(objectSeqOut)

        seqOut = torch.cat((gazeSeqOut, headSeqOut), 1)
        seqOut = torch.cat((seqOut, objectSeqOut), 1)
        
        saliencyMap = saliencyMap.reshape(saliencyMap.size(0), self.saliencyNum, self.saliencyWidth, self.saliencyWidth)
        saliencyFeatures = self.SalCNN(saliencyMap)
        saliencyOut = saliencyFeatures.reshape(saliencyMap.size(0), -1)        
        saliencyOut = self.SalCNNDropout(saliencyOut)
        
        prdInput = torch.cat((seqOut, saliencyOut), 1)
        prdOut = self.PrdFC(prdInput)
        out = prdOut.mm(self.cluster)
        #out = out + currentGaze
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out  

    
# FixationNet for forecasting eye fixations on DGaze dataset.
# CurrentGaze is not used.
class FixationNet_DGazeDataset(nn.Module):
    def __init__(self, gazeSeqSize, headSeqSize, objectSeqSize, saliencySize, clusterPath):
        super().__init__()
        # the input params
        self.gazeSeqSize = gazeSeqSize
        self.headSeqSize = headSeqSize
        self.objectSeqSize = objectSeqSize
        self.saliencySize = saliencySize
        
        print('gazeSeqSize: {}'.format(self.gazeSeqSize))
        print('headSeqSize: {}'.format(self.headSeqSize))
        print('taskObjectSeqSize: {}'.format(self.objectSeqSize))
        print('saliencySize: {}'.format(self.saliencySize))
        
        # preset params
        self.gazeFeatureNum = 2
        self.gazeSeqLength = int(self.gazeSeqSize/self.gazeFeatureNum)
        self.headFeatureNum = 2
        self.headSeqLength = int(self.headSeqSize/self.headFeatureNum)
        # the nearest 3 dynamic objects, 3*3
        self.objectFeatureNum = 9
        self.objectSeqLength = int(self.objectSeqSize/self.objectFeatureNum)
        self.saliencyWidth = 24
        self.saliencyNum = int(self.saliencySize/(self.saliencyWidth*self.saliencyWidth))
        
        # load the cluster centers
        self.cluster = torch.from_numpy(np.load(clusterPath)).float().to(device)
        self.clusterSize = self.cluster.shape[0]
        print('Cluster Size: {}'.format(self.clusterSize))
        
        
        # GazeSeqCNN1D Module
        gazeSeqCNN1D_outChannels1 = 32
        gazeSeqCNN1D_poolingRate1 = 2
        gazeSeqCNN1D_kernelSize1 = 1
        gazeSeqCNN1D_featureSize1 = floor((self.gazeSeqLength - gazeSeqCNN1D_kernelSize1 + 1)/gazeSeqCNN1D_poolingRate1)
        self.gazeSeqCNN1D_outputSize = gazeSeqCNN1D_featureSize1 * gazeSeqCNN1D_outChannels1
        self.gazeSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.gazeFeatureNum, out_channels=gazeSeqCNN1D_outChannels1,kernel_size=gazeSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(gazeSeqCNN1D_outChannels1),
            #nn.ReLU(),
            #nn.LeakyReLU(),
            nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(gazeSeqCNN1D_poolingRate1),
             )
        
        gazeSeqCNN_dropoutRate = 0.0
        self.gazeSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = gazeSeqCNN_dropoutRate)
             )       
        
        # HeadSeqCNN1D Module
        headSeqCNN1D_outChannels1 = 64
        headSeqCNN1D_poolingRate1 = 2
        headSeqCNN1D_kernelSize1 = 1
        headSeqCNN1D_featureSize1 = floor((self.headSeqLength - headSeqCNN1D_kernelSize1 + 1)/headSeqCNN1D_poolingRate1)
        self.headSeqCNN1D_outputSize = headSeqCNN1D_featureSize1 * headSeqCNN1D_outChannels1
        self.headSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.headFeatureNum, out_channels=headSeqCNN1D_outChannels1,kernel_size=headSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(headSeqCNN1D_outChannels1),
            #nn.ReLU(),
            #nn.LeakyReLU(),
            nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(headSeqCNN1D_poolingRate1),
            #nn.Conv1d(in_channels=headSeqCNN1D_outChannels1, out_channels=headSeqCNN1D_outChannels2,kernel_size=headSeqCNN1D_kernelSize2),
            #nn.BatchNorm1d(headSeqCNN1D_outChannels2),
            #nn.Sigmoid(),
            #nn.MaxPool1d(headSeqCNN1D_poolingRate2),            
             )
        
        headSeqCNN_dropoutRate = 0.0
        self.headSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = headSeqCNN_dropoutRate)
             )       
        
        
        # objectSeqCNN1D Module
        objectSeqCNN1D_outChannels1 = 64
        objectSeqCNN1D_poolingRate1 = 2
        objectSeqCNN1D_kernelSize1 = 1
        objectSeqCNN1D_featureSize1 = floor((self.objectSeqLength - objectSeqCNN1D_kernelSize1 + 1)/objectSeqCNN1D_poolingRate1)
        objectSeqCNN1D_outChannels2 = 32
        objectSeqCNN1D_poolingRate2 = 2
        objectSeqCNN1D_kernelSize2 = 1
        objectSeqCNN1D_featureSize2 = floor((objectSeqCNN1D_featureSize1 - objectSeqCNN1D_kernelSize2 + 1)/objectSeqCNN1D_poolingRate2)
        self.objectSeqCNN1D_outputSize = objectSeqCNN1D_featureSize2 * objectSeqCNN1D_outChannels2
        self.objectSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.objectFeatureNum, out_channels=objectSeqCNN1D_outChannels1,kernel_size=objectSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(objectSeqCNN1D_outChannels1),
            #nn.ReLU(),
            nn.Sigmoid(),
            #nn.LeakyReLU(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(objectSeqCNN1D_poolingRate1),
            nn.Conv1d(in_channels=objectSeqCNN1D_outChannels1, out_channels=objectSeqCNN1D_outChannels2,kernel_size=objectSeqCNN1D_kernelSize2),
            nn.BatchNorm1d(objectSeqCNN1D_outChannels2),
            #nn.ReLU(),
            nn.Sigmoid(),
            nn.MaxPool1d(objectSeqCNN1D_poolingRate2),
             )
        
        objectSeqCNN_dropoutRate = 0.0
        self.objectSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = objectSeqCNN_dropoutRate)
             )       
        
         
        # SalCNN Module
        salCNN_outChannels1 = 8
        salCNN_poolingRate1 = 2
        salCNN_kernelSize1 = 1
        salCNN_padding1 = int((salCNN_kernelSize1-1)/2)
        salCNN_imageSize1 = floor((self.saliencyWidth - salCNN_kernelSize1 + 2*salCNN_padding1 + 1)/salCNN_poolingRate1)
        self.salCNN_outputSize = salCNN_imageSize1 * salCNN_imageSize1 * salCNN_outChannels1        
        self.SalCNN = nn.Sequential(
            nn.Conv2d(self.saliencyNum, salCNN_outChannels1, kernel_size=salCNN_kernelSize1, padding=salCNN_padding1),
            nn.BatchNorm2d(salCNN_outChannels1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool2d(kernel_size=salCNN_poolingRate1),
             )
        
        salCNN_dropoutRate = 0.5
        self.SalCNNDropout = nn.Sequential(
            nn.Dropout(p = salCNN_dropoutRate)
             )       
        
        
        # PrdFC Module
        prdFC_inputSize = self.gazeSeqCNN1D_outputSize + self.headSeqCNN1D_outputSize + self.objectSeqCNN1D_outputSize + self.salCNN_outputSize
        prdFC_linearSize1 = 128
        prdFC_linearSize2 = self.clusterSize
        prdFC_dropoutRate = 0.5
        # the prediction fc layer
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            #nn.ReLU(),
            nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.Softmax(dim = 1),
             )
        
        
    def forward1(self, x):
        #currentGaze = x[:, 0:2]
        index = self.gazeSeqSize
        gazeSeq = x[:, 0:index]
        headSeq = x[:, index: index+self.headSeqSize]
        index += self.headSeqSize
        objectSeq = x[:, index: index+self.objectSeqSize]
        index += self.objectSeqSize
        saliencyMap = x[:, index: index+self.saliencySize]
        
        gazeSeq = gazeSeq.reshape(-1, self.gazeSeqLength, self.gazeFeatureNum)
        gazeSeq = gazeSeq.permute(0,2,1)        
        gazeSeqOut = self.gazeSeqCNN1D(gazeSeq)
        gazeSeqOut = gazeSeqOut.reshape(-1, self.gazeSeqCNN1D_outputSize)
        gazeSeqOut = self.gazeSeqCNNDropout(gazeSeqOut)
        
        headSeq = headSeq.reshape(-1, self.headSeqLength, self.headFeatureNum)
        headSeq = headSeq.permute(0,2,1)        
        headSeqOut = self.headSeqCNN1D(headSeq)
        headSeqOut = headSeqOut.reshape(-1, self.headSeqCNN1D_outputSize)
        headSeqOut = self.headSeqCNNDropout(headSeqOut)
        
        objectSeq = objectSeq.reshape(-1, self.objectSeqLength, self.objectFeatureNum)
        objectSeq = objectSeq.permute(0,2,1)        
        objectSeqOut = self.objectSeqCNN1D(objectSeq)
        objectSeqOut = objectSeqOut.reshape(-1, self.objectSeqCNN1D_outputSize)
        objectSeqOut = self.objectSeqCNNDropout(objectSeqOut)

        seqOut = torch.cat((gazeSeqOut, headSeqOut), 1)
        seqOut = torch.cat((seqOut, objectSeqOut), 1)
        
        saliencyMap = saliencyMap.reshape(saliencyMap.size(0), self.saliencyNum, self.saliencyWidth, self.saliencyWidth)
        saliencyFeatures = self.SalCNN(saliencyMap)
        saliencyOut = saliencyFeatures.reshape(saliencyMap.size(0), -1)        
        saliencyOut = self.SalCNNDropout(saliencyOut)
        
        prdInput = torch.cat((seqOut, saliencyOut), 1)
        prdOut = self.PrdFC(prdInput)
        out = prdOut.mm(self.cluster)
        #out = out + currentGaze
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out  

    
class FixationNet_without_Saliency(nn.Module):
    def __init__(self, gazeSeqSize, headSeqSize, objectSeqSize, saliencySize, clusterPath):
        super().__init__()
        # the input params
        self.gazeSeqSize = gazeSeqSize
        self.headSeqSize = headSeqSize
        self.objectSeqSize = objectSeqSize
        self.saliencySize = saliencySize
        
        print('gazeSeqSize: {}'.format(self.gazeSeqSize))
        print('headSeqSize: {}'.format(self.headSeqSize))
        print('taskObjectSeqSize: {}'.format(self.objectSeqSize))
        print('saliencySize: {}'.format(self.saliencySize))
        
        # preset params
        self.gazeFeatureNum = 2
        self.gazeSeqLength = int(self.gazeSeqSize/self.gazeFeatureNum)
        self.headFeatureNum = 2
        self.headSeqLength = int(self.headSeqSize/self.headFeatureNum)
        self.objectFeatureNum = 12
        self.objectSeqLength = int(self.objectSeqSize/self.objectFeatureNum)
        self.saliencyWidth = 24
        self.saliencyNum = int(self.saliencySize/(self.saliencyWidth*self.saliencyWidth))
        
        # load the cluster centers
        self.cluster = torch.from_numpy(np.load(clusterPath)).float().to(device)
        self.clusterSize = self.cluster.shape[0]
        print('Cluster Size: {}'.format(self.clusterSize))
        
        
        # GazeSeqCNN1D Module
        gazeSeqCNN1D_outChannels1 = 32
        gazeSeqCNN1D_poolingRate1 = 2
        gazeSeqCNN1D_kernelSize1 = 1
        gazeSeqCNN1D_featureSize1 = floor((self.gazeSeqLength - gazeSeqCNN1D_kernelSize1 + 1)/gazeSeqCNN1D_poolingRate1)
        self.gazeSeqCNN1D_outputSize = gazeSeqCNN1D_featureSize1 * gazeSeqCNN1D_outChannels1
        self.gazeSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.gazeFeatureNum, out_channels=gazeSeqCNN1D_outChannels1,kernel_size=gazeSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(gazeSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.LeakyReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(gazeSeqCNN1D_poolingRate1),
             )
        
        gazeSeqCNN_dropoutRate = 0.0
        self.gazeSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = gazeSeqCNN_dropoutRate)
             )       
        
        # HeadSeqCNN1D Module
        headSeqCNN1D_outChannels1 = 64
        headSeqCNN1D_poolingRate1 = 2
        headSeqCNN1D_kernelSize1 = 1
        headSeqCNN1D_featureSize1 = floor((self.headSeqLength - headSeqCNN1D_kernelSize1 + 1)/headSeqCNN1D_poolingRate1)
        self.headSeqCNN1D_outputSize = headSeqCNN1D_featureSize1 * headSeqCNN1D_outChannels1
        self.headSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.headFeatureNum, out_channels=headSeqCNN1D_outChannels1,kernel_size=headSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(headSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.LeakyReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(headSeqCNN1D_poolingRate1),
            #nn.Conv1d(in_channels=headSeqCNN1D_outChannels1, out_channels=headSeqCNN1D_outChannels2,kernel_size=headSeqCNN1D_kernelSize2),
            #nn.BatchNorm1d(headSeqCNN1D_outChannels2),
            #nn.Sigmoid(),
            #nn.MaxPool1d(headSeqCNN1D_poolingRate2),            
             )
        
        headSeqCNN_dropoutRate = 0.0
        self.headSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = headSeqCNN_dropoutRate)
             )       
        
        
        # objectSeqCNN1D Module
        objectSeqCNN1D_outChannels1 = 64
        objectSeqCNN1D_poolingRate1 = 2
        objectSeqCNN1D_kernelSize1 = 1
        objectSeqCNN1D_featureSize1 = floor((self.objectSeqLength - objectSeqCNN1D_kernelSize1 + 1)/objectSeqCNN1D_poolingRate1)
        objectSeqCNN1D_outChannels2 = 32
        objectSeqCNN1D_poolingRate2 = 2
        objectSeqCNN1D_kernelSize2 = 1
        objectSeqCNN1D_featureSize2 = floor((objectSeqCNN1D_featureSize1 - objectSeqCNN1D_kernelSize2 + 1)/objectSeqCNN1D_poolingRate2)
        self.objectSeqCNN1D_outputSize = objectSeqCNN1D_featureSize2 * objectSeqCNN1D_outChannels2
        self.objectSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.objectFeatureNum, out_channels=objectSeqCNN1D_outChannels1,kernel_size=objectSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(objectSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.LeakyReLU(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(objectSeqCNN1D_poolingRate1),
            nn.Conv1d(in_channels=objectSeqCNN1D_outChannels1, out_channels=objectSeqCNN1D_outChannels2,kernel_size=objectSeqCNN1D_kernelSize2),
            nn.BatchNorm1d(objectSeqCNN1D_outChannels2),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.MaxPool1d(objectSeqCNN1D_poolingRate2),
             )
        
        objectSeqCNN_dropoutRate = 0.0
        self.objectSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = objectSeqCNN_dropoutRate)
             )       
        
         
        # SalCNN Module
        salCNN_outChannels1 = 8
        salCNN_poolingRate1 = 2
        salCNN_kernelSize1 = 1
        salCNN_padding1 = int((salCNN_kernelSize1-1)/2)
        salCNN_imageSize1 = floor((self.saliencyWidth - salCNN_kernelSize1 + 2*salCNN_padding1 + 1)/salCNN_poolingRate1)
        self.salCNN_outputSize = salCNN_imageSize1 * salCNN_imageSize1 * salCNN_outChannels1        
        self.SalCNN = nn.Sequential(
            nn.Conv2d(self.saliencyNum, salCNN_outChannels1, kernel_size=salCNN_kernelSize1, padding=salCNN_padding1),
            nn.BatchNorm2d(salCNN_outChannels1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool2d(kernel_size=salCNN_poolingRate1),
             )
        
        salCNN_dropoutRate = 0.5
        self.SalCNNDropout = nn.Sequential(
            nn.Dropout(p = salCNN_dropoutRate)
             )       
        
        
        # PrdFC Module
        prdFC_inputSize = self.gazeSeqCNN1D_outputSize + self.headSeqCNN1D_outputSize + self.objectSeqCNN1D_outputSize
        prdFC_linearSize1 = 128
        #prdFC_linearSize2 = 128
        prdFC_linearSize2 = self.clusterSize
        prdFC_dropoutRate = 0.5
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2), 
            #nn.BatchNorm1d(prdFC_linearSize2),
            #nn.Sigmoid(),
            #nn.Dropout(p = prdFC_dropoutRate),
            #nn.Linear(prdFC_linearSize2, 2),
            nn.Softmax(dim = 1),
             )
        
        
    def forward1(self, x):
        currentGaze = x[:, 0:2]
        index = self.gazeSeqSize
        gazeSeq = x[:, 0:index]
        headSeq = x[:, index: index+self.headSeqSize]
        index += self.headSeqSize
        objectSeq = x[:, index: index+self.objectSeqSize]
        index += self.objectSeqSize
        #saliencyMap = x[:, index: index+self.saliencySize]
        
        gazeSeq = gazeSeq.reshape(-1, self.gazeSeqLength, self.gazeFeatureNum)
        gazeSeq = gazeSeq.permute(0,2,1)        
        gazeSeqOut = self.gazeSeqCNN1D(gazeSeq)
        gazeSeqOut = gazeSeqOut.reshape(-1, self.gazeSeqCNN1D_outputSize)
        gazeSeqOut = self.gazeSeqCNNDropout(gazeSeqOut)
        
        headSeq = headSeq.reshape(-1, self.headSeqLength, self.headFeatureNum)
        headSeq = headSeq.permute(0,2,1)        
        headSeqOut = self.headSeqCNN1D(headSeq)
        headSeqOut = headSeqOut.reshape(-1, self.headSeqCNN1D_outputSize)
        headSeqOut = self.headSeqCNNDropout(headSeqOut)
        
        objectSeq = objectSeq.reshape(-1, self.objectSeqLength, self.objectFeatureNum)
        objectSeq = objectSeq.permute(0,2,1)        
        objectSeqOut = self.objectSeqCNN1D(objectSeq)
        objectSeqOut = objectSeqOut.reshape(-1, self.objectSeqCNN1D_outputSize)
        objectSeqOut = self.objectSeqCNNDropout(objectSeqOut)

        seqOut = torch.cat((gazeSeqOut, headSeqOut), 1)
        seqOut = torch.cat((seqOut, objectSeqOut), 1)
        
        #saliencyMap = saliencyMap.reshape(saliencyMap.size(0), self.saliencyNum, self.saliencyWidth, self.saliencyWidth)
        #saliencyFeatures = self.SalCNN(saliencyMap)
        #saliencyOut = saliencyFeatures.reshape(saliencyMap.size(0), -1)        
        #saliencyOut = self.SalCNNDropout(saliencyOut)
        
        prdInput = seqOut
        prdOut = self.PrdFC(prdInput)
        out = prdOut.mm(self.cluster)
        out = out + currentGaze
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out  
    
    
class FixationNet_without_Task(nn.Module):
    def __init__(self, gazeSeqSize, headSeqSize, objectSeqSize, saliencySize, clusterPath):
        super().__init__()
        # the input params
        self.gazeSeqSize = gazeSeqSize
        self.headSeqSize = headSeqSize
        self.objectSeqSize = objectSeqSize
        self.saliencySize = saliencySize
        
        print('gazeSeqSize: {}'.format(self.gazeSeqSize))
        print('headSeqSize: {}'.format(self.headSeqSize))
        print('taskObjectSeqSize: {}'.format(self.objectSeqSize))
        print('saliencySize: {}'.format(self.saliencySize))
        
        # preset params
        self.gazeFeatureNum = 2
        self.gazeSeqLength = int(self.gazeSeqSize/self.gazeFeatureNum)
        self.headFeatureNum = 2
        self.headSeqLength = int(self.headSeqSize/self.headFeatureNum)
        self.objectFeatureNum = 12
        self.objectSeqLength = int(self.objectSeqSize/self.objectFeatureNum)
        self.saliencyWidth = 24
        self.saliencyNum = int(self.saliencySize/(self.saliencyWidth*self.saliencyWidth))
        
        # load the cluster centers
        self.cluster = torch.from_numpy(np.load(clusterPath)).float().to(device)
        self.clusterSize = self.cluster.shape[0]
        print('Cluster Size: {}'.format(self.clusterSize))
        
        
        # GazeSeqCNN1D Module
        gazeSeqCNN1D_outChannels1 = 32
        gazeSeqCNN1D_poolingRate1 = 2
        gazeSeqCNN1D_kernelSize1 = 1
        gazeSeqCNN1D_featureSize1 = floor((self.gazeSeqLength - gazeSeqCNN1D_kernelSize1 + 1)/gazeSeqCNN1D_poolingRate1)
        self.gazeSeqCNN1D_outputSize = gazeSeqCNN1D_featureSize1 * gazeSeqCNN1D_outChannels1
        self.gazeSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.gazeFeatureNum, out_channels=gazeSeqCNN1D_outChannels1,kernel_size=gazeSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(gazeSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.LeakyReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(gazeSeqCNN1D_poolingRate1),
             )
        
        gazeSeqCNN_dropoutRate = 0.0
        self.gazeSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = gazeSeqCNN_dropoutRate)
             )       
        
        # HeadSeqCNN1D Module
        headSeqCNN1D_outChannels1 = 64
        headSeqCNN1D_poolingRate1 = 2
        headSeqCNN1D_kernelSize1 = 1
        headSeqCNN1D_featureSize1 = floor((self.headSeqLength - headSeqCNN1D_kernelSize1 + 1)/headSeqCNN1D_poolingRate1)
        self.headSeqCNN1D_outputSize = headSeqCNN1D_featureSize1 * headSeqCNN1D_outChannels1
        self.headSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.headFeatureNum, out_channels=headSeqCNN1D_outChannels1,kernel_size=headSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(headSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.LeakyReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(headSeqCNN1D_poolingRate1),
            #nn.Conv1d(in_channels=headSeqCNN1D_outChannels1, out_channels=headSeqCNN1D_outChannels2,kernel_size=headSeqCNN1D_kernelSize2),
            #nn.BatchNorm1d(headSeqCNN1D_outChannels2),
            #nn.Sigmoid(),
            #nn.MaxPool1d(headSeqCNN1D_poolingRate2),            
             )
        
        headSeqCNN_dropoutRate = 0.0
        self.headSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = headSeqCNN_dropoutRate)
             )       

        
        # objectSeqCNN1D Module
        objectSeqCNN1D_outChannels1 = 64
        objectSeqCNN1D_poolingRate1 = 2
        objectSeqCNN1D_kernelSize1 = 1
        objectSeqCNN1D_featureSize1 = floor((self.objectSeqLength - objectSeqCNN1D_kernelSize1 + 1)/objectSeqCNN1D_poolingRate1)
        objectSeqCNN1D_outChannels2 = 32
        objectSeqCNN1D_poolingRate2 = 2
        objectSeqCNN1D_kernelSize2 = 1
        objectSeqCNN1D_featureSize2 = floor((objectSeqCNN1D_featureSize1 - objectSeqCNN1D_kernelSize2 + 1)/objectSeqCNN1D_poolingRate2)
        self.objectSeqCNN1D_outputSize = objectSeqCNN1D_featureSize2 * objectSeqCNN1D_outChannels2
        self.objectSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.objectFeatureNum, out_channels=objectSeqCNN1D_outChannels1,kernel_size=objectSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(objectSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.LeakyReLU(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(objectSeqCNN1D_poolingRate1),
            nn.Conv1d(in_channels=objectSeqCNN1D_outChannels1, out_channels=objectSeqCNN1D_outChannels2,kernel_size=objectSeqCNN1D_kernelSize2),
            nn.BatchNorm1d(objectSeqCNN1D_outChannels2),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.MaxPool1d(objectSeqCNN1D_poolingRate2),
             )
        
        objectSeqCNN_dropoutRate = 0.0
        self.objectSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = objectSeqCNN_dropoutRate)
             )       
        
         
        # SalCNN Module
        salCNN_outChannels1 = 8
        salCNN_poolingRate1 = 2
        salCNN_kernelSize1 = 1
        salCNN_padding1 = int((salCNN_kernelSize1-1)/2)
        salCNN_imageSize1 = floor((self.saliencyWidth - salCNN_kernelSize1 + 2*salCNN_padding1 + 1)/salCNN_poolingRate1)
        self.salCNN_outputSize = salCNN_imageSize1 * salCNN_imageSize1 * salCNN_outChannels1        
        self.SalCNN = nn.Sequential(
            nn.Conv2d(self.saliencyNum, salCNN_outChannels1, kernel_size=salCNN_kernelSize1, padding=salCNN_padding1),
            nn.BatchNorm2d(salCNN_outChannels1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool2d(kernel_size=salCNN_poolingRate1),
             )
        
        salCNN_dropoutRate = 0.5
        self.SalCNNDropout = nn.Sequential(
            nn.Dropout(p = salCNN_dropoutRate)
             )       
        
        
        # PrdFC Module
        prdFC_inputSize = self.gazeSeqCNN1D_outputSize + self.headSeqCNN1D_outputSize + self.salCNN_outputSize
        prdFC_linearSize1 = 128
        prdFC_linearSize2 = self.clusterSize
        prdFC_dropoutRate = 0.5
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.Softmax(dim = 1),
             )
        
        
    def forward1(self, x):
        currentGaze = x[:, 0:2]
        index = self.gazeSeqSize
        gazeSeq = x[:, 0:index]
        headSeq = x[:, index: index+self.headSeqSize]
        index += self.headSeqSize
        #objectSeq = x[:, index: index+self.objectSeqSize]
        index += self.objectSeqSize
        saliencyMap = x[:, index: index+self.saliencySize]
        
        gazeSeq = gazeSeq.reshape(-1, self.gazeSeqLength, self.gazeFeatureNum)
        gazeSeq = gazeSeq.permute(0,2,1)        
        gazeSeqOut = self.gazeSeqCNN1D(gazeSeq)
        gazeSeqOut = gazeSeqOut.reshape(-1, self.gazeSeqCNN1D_outputSize)
        gazeSeqOut = self.gazeSeqCNNDropout(gazeSeqOut)
        
        headSeq = headSeq.reshape(-1, self.headSeqLength, self.headFeatureNum)
        headSeq = headSeq.permute(0,2,1)        
        headSeqOut = self.headSeqCNN1D(headSeq)
        headSeqOut = headSeqOut.reshape(-1, self.headSeqCNN1D_outputSize)
        headSeqOut = self.headSeqCNNDropout(headSeqOut)
        
        #objectSeq = objectSeq.reshape(-1, self.objectSeqLength, self.objectFeatureNum)
        #objectSeq = objectSeq.permute(0,2,1)        
        #objectSeqOut = self.objectSeqCNN1D(objectSeq)
        #objectSeqOut = objectSeqOut.reshape(-1, self.objectSeqCNN1D_outputSize)
        #objectSeqOut = self.objectSeqCNNDropout(objectSeqOut)

        seqOut = torch.cat((gazeSeqOut, headSeqOut), 1)
        
        saliencyMap = saliencyMap.reshape(saliencyMap.size(0), self.saliencyNum, self.saliencyWidth, self.saliencyWidth)
        saliencyFeatures = self.SalCNN(saliencyMap)
        saliencyOut = saliencyFeatures.reshape(saliencyMap.size(0), -1)        
        saliencyOut = self.SalCNNDropout(saliencyOut)
        
        prdInput = torch.cat((seqOut, saliencyOut), 1)
        prdOut = self.PrdFC(prdInput)
        out = prdOut.mm(self.cluster)
        out = out + currentGaze
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out  
    

class FixationNet_without_Gaze(nn.Module):
    def __init__(self, gazeSeqSize, headSeqSize, objectSeqSize, saliencySize, clusterPath):
        super().__init__()
        # the input params
        self.gazeSeqSize = gazeSeqSize
        self.headSeqSize = headSeqSize
        self.objectSeqSize = objectSeqSize
        self.saliencySize = saliencySize
        
        print('gazeSeqSize: {}'.format(self.gazeSeqSize))
        print('headSeqSize: {}'.format(self.headSeqSize))
        print('taskObjectSeqSize: {}'.format(self.objectSeqSize))
        print('saliencySize: {}'.format(self.saliencySize))
        
        # preset params
        self.gazeFeatureNum = 2
        self.gazeSeqLength = int(self.gazeSeqSize/self.gazeFeatureNum)
        self.headFeatureNum = 2
        self.headSeqLength = int(self.headSeqSize/self.headFeatureNum)
        self.objectFeatureNum = 12
        self.objectSeqLength = int(self.objectSeqSize/self.objectFeatureNum)
        self.saliencyWidth = 24
        self.saliencyNum = int(self.saliencySize/(self.saliencyWidth*self.saliencyWidth))
        
        # load the cluster centers
        self.cluster = torch.from_numpy(np.load(clusterPath)).float().to(device)
        self.clusterSize = self.cluster.shape[0]
        print('Cluster Size: {}'.format(self.clusterSize))
        
        
        # GazeSeqCNN1D Module
        gazeSeqCNN1D_outChannels1 = 32
        gazeSeqCNN1D_poolingRate1 = 2
        gazeSeqCNN1D_kernelSize1 = 1
        gazeSeqCNN1D_featureSize1 = floor((self.gazeSeqLength - gazeSeqCNN1D_kernelSize1 + 1)/gazeSeqCNN1D_poolingRate1)
        self.gazeSeqCNN1D_outputSize = gazeSeqCNN1D_featureSize1 * gazeSeqCNN1D_outChannels1
        self.gazeSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.gazeFeatureNum, out_channels=gazeSeqCNN1D_outChannels1,kernel_size=gazeSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(gazeSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.LeakyReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(gazeSeqCNN1D_poolingRate1),
             )
        
        gazeSeqCNN_dropoutRate = 0.0
        self.gazeSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = gazeSeqCNN_dropoutRate)
             )       
        
        # HeadSeqCNN1D Module
        headSeqCNN1D_outChannels1 = 64
        headSeqCNN1D_poolingRate1 = 2
        headSeqCNN1D_kernelSize1 = 1
        headSeqCNN1D_featureSize1 = floor((self.headSeqLength - headSeqCNN1D_kernelSize1 + 1)/headSeqCNN1D_poolingRate1)
        self.headSeqCNN1D_outputSize = headSeqCNN1D_featureSize1 * headSeqCNN1D_outChannels1
        self.headSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.headFeatureNum, out_channels=headSeqCNN1D_outChannels1,kernel_size=headSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(headSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.LeakyReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(headSeqCNN1D_poolingRate1),
            #nn.Conv1d(in_channels=headSeqCNN1D_outChannels1, out_channels=headSeqCNN1D_outChannels2,kernel_size=headSeqCNN1D_kernelSize2),
            #nn.BatchNorm1d(headSeqCNN1D_outChannels2),
            #nn.Sigmoid(),
            #nn.MaxPool1d(headSeqCNN1D_poolingRate2),            
             )
        
        headSeqCNN_dropoutRate = 0.0
        self.headSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = headSeqCNN_dropoutRate)
             )       

        
        # objectSeqCNN1D Module
        objectSeqCNN1D_outChannels1 = 64
        objectSeqCNN1D_poolingRate1 = 2
        objectSeqCNN1D_kernelSize1 = 1
        objectSeqCNN1D_featureSize1 = floor((self.objectSeqLength - objectSeqCNN1D_kernelSize1 + 1)/objectSeqCNN1D_poolingRate1)
        objectSeqCNN1D_outChannels2 = 32
        objectSeqCNN1D_poolingRate2 = 2
        objectSeqCNN1D_kernelSize2 = 1
        objectSeqCNN1D_featureSize2 = floor((objectSeqCNN1D_featureSize1 - objectSeqCNN1D_kernelSize2 + 1)/objectSeqCNN1D_poolingRate2)
        self.objectSeqCNN1D_outputSize = objectSeqCNN1D_featureSize2 * objectSeqCNN1D_outChannels2
        self.objectSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.objectFeatureNum, out_channels=objectSeqCNN1D_outChannels1,kernel_size=objectSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(objectSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.LeakyReLU(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(objectSeqCNN1D_poolingRate1),
            nn.Conv1d(in_channels=objectSeqCNN1D_outChannels1, out_channels=objectSeqCNN1D_outChannels2,kernel_size=objectSeqCNN1D_kernelSize2),
            nn.BatchNorm1d(objectSeqCNN1D_outChannels2),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.MaxPool1d(objectSeqCNN1D_poolingRate2),
             )
        
        objectSeqCNN_dropoutRate = 0.0
        self.objectSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = objectSeqCNN_dropoutRate)
             )       
        
         
        # SalCNN Module
        salCNN_outChannels1 = 8
        salCNN_poolingRate1 = 2
        salCNN_kernelSize1 = 1
        salCNN_padding1 = int((salCNN_kernelSize1-1)/2)
        salCNN_imageSize1 = floor((self.saliencyWidth - salCNN_kernelSize1 + 2*salCNN_padding1 + 1)/salCNN_poolingRate1)
        self.salCNN_outputSize = salCNN_imageSize1 * salCNN_imageSize1 * salCNN_outChannels1        
        self.SalCNN = nn.Sequential(
            nn.Conv2d(self.saliencyNum, salCNN_outChannels1, kernel_size=salCNN_kernelSize1, padding=salCNN_padding1),
            nn.BatchNorm2d(salCNN_outChannels1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool2d(kernel_size=salCNN_poolingRate1),
             )
        
        salCNN_dropoutRate = 0.5
        self.SalCNNDropout = nn.Sequential(
            nn.Dropout(p = salCNN_dropoutRate)
             )       
        
        
        # PrdFC Module
        prdFC_inputSize = self.headSeqCNN1D_outputSize + self.objectSeqCNN1D_outputSize + self.salCNN_outputSize
        prdFC_linearSize1 = 128
        prdFC_linearSize2 = self.clusterSize
        prdFC_dropoutRate = 0.5
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.Softmax(dim = 1),
             )
        
        
    def forward1(self, x):
        currentGaze = x[:, 0:2]
        index = self.gazeSeqSize
        #gazeSeq = x[:, 0:index]
        headSeq = x[:, index: index+self.headSeqSize]
        index += self.headSeqSize
        objectSeq = x[:, index: index+self.objectSeqSize]
        index += self.objectSeqSize
        saliencyMap = x[:, index: index+self.saliencySize]
        
        #gazeSeq = gazeSeq.reshape(-1, self.gazeSeqLength, self.gazeFeatureNum)
        #gazeSeq = gazeSeq.permute(0,2,1)        
        #gazeSeqOut = self.gazeSeqCNN1D(gazeSeq)
        #gazeSeqOut = gazeSeqOut.reshape(-1, self.gazeSeqCNN1D_outputSize)
        #gazeSeqOut = self.gazeSeqCNNDropout(gazeSeqOut)
        
        headSeq = headSeq.reshape(-1, self.headSeqLength, self.headFeatureNum)
        headSeq = headSeq.permute(0,2,1)        
        headSeqOut = self.headSeqCNN1D(headSeq)
        headSeqOut = headSeqOut.reshape(-1, self.headSeqCNN1D_outputSize)
        headSeqOut = self.headSeqCNNDropout(headSeqOut)
        
        objectSeq = objectSeq.reshape(-1, self.objectSeqLength, self.objectFeatureNum)
        objectSeq = objectSeq.permute(0,2,1)        
        objectSeqOut = self.objectSeqCNN1D(objectSeq)
        objectSeqOut = objectSeqOut.reshape(-1, self.objectSeqCNN1D_outputSize)
        objectSeqOut = self.objectSeqCNNDropout(objectSeqOut)

        seqOut = torch.cat((headSeqOut, objectSeqOut), 1)
        
        saliencyMap = saliencyMap.reshape(saliencyMap.size(0), self.saliencyNum, self.saliencyWidth, self.saliencyWidth)
        saliencyFeatures = self.SalCNN(saliencyMap)
        saliencyOut = saliencyFeatures.reshape(saliencyMap.size(0), -1)        
        saliencyOut = self.SalCNNDropout(saliencyOut)
        
        prdInput = torch.cat((seqOut, saliencyOut), 1)
        prdOut = self.PrdFC(prdInput)
        out = prdOut.mm(self.cluster)
        out = out + currentGaze
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out  
    

class FixationNet_without_Head(nn.Module):
    def __init__(self, gazeSeqSize, headSeqSize, objectSeqSize, saliencySize, clusterPath):
        super().__init__()
        # the input params
        self.gazeSeqSize = gazeSeqSize
        self.headSeqSize = headSeqSize
        self.objectSeqSize = objectSeqSize
        self.saliencySize = saliencySize
        
        print('gazeSeqSize: {}'.format(self.gazeSeqSize))
        print('headSeqSize: {}'.format(self.headSeqSize))
        print('taskObjectSeqSize: {}'.format(self.objectSeqSize))
        print('saliencySize: {}'.format(self.saliencySize))
        
        # preset params
        self.gazeFeatureNum = 2
        self.gazeSeqLength = int(self.gazeSeqSize/self.gazeFeatureNum)
        self.headFeatureNum = 2
        self.headSeqLength = int(self.headSeqSize/self.headFeatureNum)
        self.objectFeatureNum = 12
        self.objectSeqLength = int(self.objectSeqSize/self.objectFeatureNum)
        self.saliencyWidth = 24
        self.saliencyNum = int(self.saliencySize/(self.saliencyWidth*self.saliencyWidth))
        
        # load the cluster centers
        self.cluster = torch.from_numpy(np.load(clusterPath)).float().to(device)
        self.clusterSize = self.cluster.shape[0]
        print('Cluster Size: {}'.format(self.clusterSize))
        
        
        # GazeSeqCNN1D Module
        gazeSeqCNN1D_outChannels1 = 32
        gazeSeqCNN1D_poolingRate1 = 2
        gazeSeqCNN1D_kernelSize1 = 1
        gazeSeqCNN1D_featureSize1 = floor((self.gazeSeqLength - gazeSeqCNN1D_kernelSize1 + 1)/gazeSeqCNN1D_poolingRate1)
        self.gazeSeqCNN1D_outputSize = gazeSeqCNN1D_featureSize1 * gazeSeqCNN1D_outChannels1
        self.gazeSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.gazeFeatureNum, out_channels=gazeSeqCNN1D_outChannels1,kernel_size=gazeSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(gazeSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.LeakyReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(gazeSeqCNN1D_poolingRate1),
             )
        
        gazeSeqCNN_dropoutRate = 0.0
        self.gazeSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = gazeSeqCNN_dropoutRate)
             )       
        
        # HeadSeqCNN1D Module
        headSeqCNN1D_outChannels1 = 64
        headSeqCNN1D_poolingRate1 = 2
        headSeqCNN1D_kernelSize1 = 1
        headSeqCNN1D_featureSize1 = floor((self.headSeqLength - headSeqCNN1D_kernelSize1 + 1)/headSeqCNN1D_poolingRate1)
        self.headSeqCNN1D_outputSize = headSeqCNN1D_featureSize1 * headSeqCNN1D_outChannels1
        self.headSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.headFeatureNum, out_channels=headSeqCNN1D_outChannels1,kernel_size=headSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(headSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.LeakyReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(headSeqCNN1D_poolingRate1),
            #nn.Conv1d(in_channels=headSeqCNN1D_outChannels1, out_channels=headSeqCNN1D_outChannels2,kernel_size=headSeqCNN1D_kernelSize2),
            #nn.BatchNorm1d(headSeqCNN1D_outChannels2),
            #nn.Sigmoid(),
            #nn.MaxPool1d(headSeqCNN1D_poolingRate2),            
             )
        
        headSeqCNN_dropoutRate = 0.0
        self.headSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = headSeqCNN_dropoutRate)
             )       

        
        # objectSeqCNN1D Module
        objectSeqCNN1D_outChannels1 = 64
        objectSeqCNN1D_poolingRate1 = 2
        objectSeqCNN1D_kernelSize1 = 1
        objectSeqCNN1D_featureSize1 = floor((self.objectSeqLength - objectSeqCNN1D_kernelSize1 + 1)/objectSeqCNN1D_poolingRate1)
        objectSeqCNN1D_outChannels2 = 32
        objectSeqCNN1D_poolingRate2 = 2
        objectSeqCNN1D_kernelSize2 = 1
        objectSeqCNN1D_featureSize2 = floor((objectSeqCNN1D_featureSize1 - objectSeqCNN1D_kernelSize2 + 1)/objectSeqCNN1D_poolingRate2)
        self.objectSeqCNN1D_outputSize = objectSeqCNN1D_featureSize2 * objectSeqCNN1D_outChannels2
        self.objectSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.objectFeatureNum, out_channels=objectSeqCNN1D_outChannels1,kernel_size=objectSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(objectSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.LeakyReLU(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(objectSeqCNN1D_poolingRate1),
            nn.Conv1d(in_channels=objectSeqCNN1D_outChannels1, out_channels=objectSeqCNN1D_outChannels2,kernel_size=objectSeqCNN1D_kernelSize2),
            nn.BatchNorm1d(objectSeqCNN1D_outChannels2),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.MaxPool1d(objectSeqCNN1D_poolingRate2),
             )
        
        objectSeqCNN_dropoutRate = 0.0
        self.objectSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = objectSeqCNN_dropoutRate)
             )       
        
         
        # SalCNN Module
        salCNN_outChannels1 = 8
        salCNN_poolingRate1 = 2
        salCNN_kernelSize1 = 1
        salCNN_padding1 = int((salCNN_kernelSize1-1)/2)
        salCNN_imageSize1 = floor((self.saliencyWidth - salCNN_kernelSize1 + 2*salCNN_padding1 + 1)/salCNN_poolingRate1)
        self.salCNN_outputSize = salCNN_imageSize1 * salCNN_imageSize1 * salCNN_outChannels1        
        self.SalCNN = nn.Sequential(
            nn.Conv2d(self.saliencyNum, salCNN_outChannels1, kernel_size=salCNN_kernelSize1, padding=salCNN_padding1),
            nn.BatchNorm2d(salCNN_outChannels1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool2d(kernel_size=salCNN_poolingRate1),
             )
        
        salCNN_dropoutRate = 0.5
        self.SalCNNDropout = nn.Sequential(
            nn.Dropout(p = salCNN_dropoutRate)
             )       
        
        
        # PrdFC Module
        prdFC_inputSize = self.gazeSeqCNN1D_outputSize + self.objectSeqCNN1D_outputSize + self.salCNN_outputSize
        prdFC_linearSize1 = 128
        prdFC_linearSize2 = self.clusterSize
        prdFC_dropoutRate = 0.5
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2),
            nn.Softmax(dim = 1),
             )
        
        
    def forward1(self, x):
        currentGaze = x[:, 0:2]
        index = self.gazeSeqSize
        gazeSeq = x[:, 0:index]
        #headSeq = x[:, index: index+self.headSeqSize]
        index += self.headSeqSize
        objectSeq = x[:, index: index+self.objectSeqSize]
        index += self.objectSeqSize
        saliencyMap = x[:, index: index+self.saliencySize]
        
        gazeSeq = gazeSeq.reshape(-1, self.gazeSeqLength, self.gazeFeatureNum)
        gazeSeq = gazeSeq.permute(0,2,1)        
        gazeSeqOut = self.gazeSeqCNN1D(gazeSeq)
        gazeSeqOut = gazeSeqOut.reshape(-1, self.gazeSeqCNN1D_outputSize)
        gazeSeqOut = self.gazeSeqCNNDropout(gazeSeqOut)
        
        #headSeq = headSeq.reshape(-1, self.headSeqLength, self.headFeatureNum)
        #headSeq = headSeq.permute(0,2,1)        
        #headSeqOut = self.headSeqCNN1D(headSeq)
        #headSeqOut = headSeqOut.reshape(-1, self.headSeqCNN1D_outputSize)
        #headSeqOut = self.headSeqCNNDropout(headSeqOut)
        
        objectSeq = objectSeq.reshape(-1, self.objectSeqLength, self.objectFeatureNum)
        objectSeq = objectSeq.permute(0,2,1)        
        objectSeqOut = self.objectSeqCNN1D(objectSeq)
        objectSeqOut = objectSeqOut.reshape(-1, self.objectSeqCNN1D_outputSize)
        objectSeqOut = self.objectSeqCNNDropout(objectSeqOut)

        seqOut = torch.cat((gazeSeqOut, objectSeqOut), 1)
        
        saliencyMap = saliencyMap.reshape(saliencyMap.size(0), self.saliencyNum, self.saliencyWidth, self.saliencyWidth)
        saliencyFeatures = self.SalCNN(saliencyMap)
        saliencyOut = saliencyFeatures.reshape(saliencyMap.size(0), -1)        
        saliencyOut = self.SalCNNDropout(saliencyOut)
        
        prdInput = torch.cat((seqOut, saliencyOut), 1)
        prdOut = self.PrdFC(prdInput)
        out = prdOut.mm(self.cluster)
        out = out + currentGaze
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out  
    
    
class FixationNet_without_Cluster(nn.Module):
    def __init__(self, gazeSeqSize, headSeqSize, objectSeqSize, saliencySize, clusterPath):
        super().__init__()
        # the input params
        self.gazeSeqSize = gazeSeqSize
        self.headSeqSize = headSeqSize
        self.objectSeqSize = objectSeqSize
        self.saliencySize = saliencySize
        
        print('gazeSeqSize: {}'.format(self.gazeSeqSize))
        print('headSeqSize: {}'.format(self.headSeqSize))
        print('taskObjectSeqSize: {}'.format(self.objectSeqSize))
        print('saliencySize: {}'.format(self.saliencySize))
        
        # preset params
        self.gazeFeatureNum = 2
        self.gazeSeqLength = int(self.gazeSeqSize/self.gazeFeatureNum)
        self.headFeatureNum = 2
        self.headSeqLength = int(self.headSeqSize/self.headFeatureNum)
        self.objectFeatureNum = 12
        self.objectSeqLength = int(self.objectSeqSize/self.objectFeatureNum)
        self.saliencyWidth = 24
        self.saliencyNum = int(self.saliencySize/(self.saliencyWidth*self.saliencyWidth))
                
        
        # GazeSeqCNN1D Module
        gazeSeqCNN1D_outChannels1 = 32
        gazeSeqCNN1D_poolingRate1 = 2
        gazeSeqCNN1D_kernelSize1 = 1
        gazeSeqCNN1D_featureSize1 = floor((self.gazeSeqLength - gazeSeqCNN1D_kernelSize1 + 1)/gazeSeqCNN1D_poolingRate1)
        self.gazeSeqCNN1D_outputSize = gazeSeqCNN1D_featureSize1 * gazeSeqCNN1D_outChannels1
        self.gazeSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.gazeFeatureNum, out_channels=gazeSeqCNN1D_outChannels1,kernel_size=gazeSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(gazeSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.LeakyReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(gazeSeqCNN1D_poolingRate1),
             )
        
        gazeSeqCNN_dropoutRate = 0.0
        self.gazeSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = gazeSeqCNN_dropoutRate)
             )       
        
        # HeadSeqCNN1D Module
        headSeqCNN1D_outChannels1 = 64
        headSeqCNN1D_poolingRate1 = 2
        headSeqCNN1D_kernelSize1 = 1
        headSeqCNN1D_featureSize1 = floor((self.headSeqLength - headSeqCNN1D_kernelSize1 + 1)/headSeqCNN1D_poolingRate1)
        self.headSeqCNN1D_outputSize = headSeqCNN1D_featureSize1 * headSeqCNN1D_outChannels1
        self.headSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.headFeatureNum, out_channels=headSeqCNN1D_outChannels1,kernel_size=headSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(headSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.LeakyReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(headSeqCNN1D_poolingRate1),
            #nn.Conv1d(in_channels=headSeqCNN1D_outChannels1, out_channels=headSeqCNN1D_outChannels2,kernel_size=headSeqCNN1D_kernelSize2),
            #nn.BatchNorm1d(headSeqCNN1D_outChannels2),
            #nn.Sigmoid(),
            #nn.MaxPool1d(headSeqCNN1D_poolingRate2),            
             )
        
        headSeqCNN_dropoutRate = 0.0
        self.headSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = headSeqCNN_dropoutRate)
             )       

        
        # objectSeqCNN1D Module
        objectSeqCNN1D_outChannels1 = 64
        objectSeqCNN1D_poolingRate1 = 2
        objectSeqCNN1D_kernelSize1 = 1
        objectSeqCNN1D_featureSize1 = floor((self.objectSeqLength - objectSeqCNN1D_kernelSize1 + 1)/objectSeqCNN1D_poolingRate1)
        objectSeqCNN1D_outChannels2 = 32
        objectSeqCNN1D_poolingRate2 = 2
        objectSeqCNN1D_kernelSize2 = 1
        objectSeqCNN1D_featureSize2 = floor((objectSeqCNN1D_featureSize1 - objectSeqCNN1D_kernelSize2 + 1)/objectSeqCNN1D_poolingRate2)
        self.objectSeqCNN1D_outputSize = objectSeqCNN1D_featureSize2 * objectSeqCNN1D_outChannels2
        self.objectSeqCNN1D = nn.Sequential(
            nn.Conv1d(in_channels=self.objectFeatureNum, out_channels=objectSeqCNN1D_outChannels1,kernel_size=objectSeqCNN1D_kernelSize1),
            nn.BatchNorm1d(objectSeqCNN1D_outChannels1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.LeakyReLU(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool1d(objectSeqCNN1D_poolingRate1),
            nn.Conv1d(in_channels=objectSeqCNN1D_outChannels1, out_channels=objectSeqCNN1D_outChannels2,kernel_size=objectSeqCNN1D_kernelSize2),
            nn.BatchNorm1d(objectSeqCNN1D_outChannels2),
            nn.ReLU(),
            #nn.Sigmoid(),
            nn.MaxPool1d(objectSeqCNN1D_poolingRate2),
             )
        
        objectSeqCNN_dropoutRate = 0.0
        self.objectSeqCNNDropout = nn.Sequential(
            nn.Dropout(p = objectSeqCNN_dropoutRate)
             )       
        
         
        # SalCNN Module
        salCNN_outChannels1 = 8
        salCNN_poolingRate1 = 2
        salCNN_kernelSize1 = 1
        salCNN_padding1 = int((salCNN_kernelSize1-1)/2)
        salCNN_imageSize1 = floor((self.saliencyWidth - salCNN_kernelSize1 + 2*salCNN_padding1 + 1)/salCNN_poolingRate1)
        self.salCNN_outputSize = salCNN_imageSize1 * salCNN_imageSize1 * salCNN_outChannels1        
        self.SalCNN = nn.Sequential(
            nn.Conv2d(self.saliencyNum, salCNN_outChannels1, kernel_size=salCNN_kernelSize1, padding=salCNN_padding1),
            nn.BatchNorm2d(salCNN_outChannels1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            nn.MaxPool2d(kernel_size=salCNN_poolingRate1),
             )
        
        salCNN_dropoutRate = 0.5
        self.SalCNNDropout = nn.Sequential(
            nn.Dropout(p = salCNN_dropoutRate)
             )       
        
        
        # PrdFC Module
        prdFC_inputSize = self.gazeSeqCNN1D_outputSize + self.headSeqCNN1D_outputSize + self.objectSeqCNN1D_outputSize + self.salCNN_outputSize
        prdFC_linearSize1 = 128
        prdFC_linearSize2 = 2
        #prdFC_dropoutRate = 0.0
        self.PrdFC = nn.Sequential(
            nn.Linear(prdFC_inputSize, prdFC_linearSize1),
            nn.BatchNorm1d(prdFC_linearSize1),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Tanh(),
            #nn.Softplus(),
            #nn.Dropout(p = prdFC_dropoutRate),
            nn.Linear(prdFC_linearSize1, prdFC_linearSize2)
             )
        
        
    def forward1(self, x):
        index = self.gazeSeqSize
        gazeSeq = x[:, 0:index]
        headSeq = x[:, index: index+self.headSeqSize]
        index += self.headSeqSize
        objectSeq = x[:, index: index+self.objectSeqSize]
        index += self.objectSeqSize
        saliencyMap = x[:, index: index+self.saliencySize]
        
        gazeSeq = gazeSeq.reshape(-1, self.gazeSeqLength, self.gazeFeatureNum)
        gazeSeq = gazeSeq.permute(0,2,1)        
        gazeSeqOut = self.gazeSeqCNN1D(gazeSeq)
        gazeSeqOut = gazeSeqOut.reshape(-1, self.gazeSeqCNN1D_outputSize)
        gazeSeqOut = self.gazeSeqCNNDropout(gazeSeqOut)
        
        headSeq = headSeq.reshape(-1, self.headSeqLength, self.headFeatureNum)
        headSeq = headSeq.permute(0,2,1)        
        headSeqOut = self.headSeqCNN1D(headSeq)
        headSeqOut = headSeqOut.reshape(-1, self.headSeqCNN1D_outputSize)
        headSeqOut = self.headSeqCNNDropout(headSeqOut)
        
        objectSeq = objectSeq.reshape(-1, self.objectSeqLength, self.objectFeatureNum)
        objectSeq = objectSeq.permute(0,2,1)        
        objectSeqOut = self.objectSeqCNN1D(objectSeq)
        objectSeqOut = objectSeqOut.reshape(-1, self.objectSeqCNN1D_outputSize)
        objectSeqOut = self.objectSeqCNNDropout(objectSeqOut)

        seqOut = torch.cat((gazeSeqOut, headSeqOut), 1)
        seqOut = torch.cat((seqOut, objectSeqOut), 1)
        
        saliencyMap = saliencyMap.reshape(saliencyMap.size(0), self.saliencyNum, self.saliencyWidth, self.saliencyWidth)
        saliencyFeatures = self.SalCNN(saliencyMap)
        saliencyOut = saliencyFeatures.reshape(saliencyMap.size(0), -1)        
        saliencyOut = self.SalCNNDropout(saliencyOut)
        
        prdInput = torch.cat((seqOut, saliencyOut), 1)
        prdOut = self.PrdFC(prdInput)
        out = prdOut
        return out
            
    def forward(self, x):
        out = self.forward1(x)
        return out  
    
