# Copyright (c) Hu Zhiming 2020/6/1 jimmyhu@pku.edu.cn All Rights Reserved.

import sys
sys.path.append('../')
from utils import CalAngularDist, LoadTrainingData, LoadTestData, RemakeDir, MakeDir, SeedTorch
from utils.Misc import AverageMeter
from models import AngularLoss
from models import weight_init
from models.FixationNetModels import *
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import time
import datetime
import argparse
import os


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set the random seed to ensure reproducibility
SeedTorch(seed=0)


def main(args):
    # Create the model.
    print('\n==> Creating the model...')
    model = FixationNet_DGazeDataset(args.gazeSeqSize, args.headSeqSize, args.taskSeqSize, args.saliencySize, args.clusterPath)
    model.apply(weight_init)
    #print('# Number of Model Parameters:', sum(param.numel() for param in model.parameters()))
    
    # print the parameters
    #for name, parameters in model.named_parameters():
        #print(name, parameters)
        
    model = torch.nn.DataParallel(model)
    if args.loss == 'AngularLoss':
        criterion = AngularLoss()
        print('\n==> Loss Function: AngularLoss')
    if args.loss == 'L1' or args.loss == 'MAE':
        criterion = nn.L1Loss()
        print('\n==> Loss Function: L1')
    if args.loss == 'MSE' or args.loss == 'L2':
        criterion = nn.MSELoss()
        print('\n==> Loss Function: L2')
    
    # train the model.
    if args.trainFlag == 1:
        # load the training data.
        train_loader = LoadTrainingData(args.datasetDir, args.batchSize)
        # optimizer and loss.
        lr = args.lr
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weightDecay)
        #optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        #optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        #optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov = True, weight_decay=args.weight_decay)
        #stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.stepLR, gamma=args.gamma)
        expLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=-1)
        # training start epoch
        startEpoch = 0
        # remake checkpoint directory
        RemakeDir(args.checkpoint)
            
        # training.
        localtime = time.asctime(time.localtime(time.time()))
        print('\nTraining starts at ' + localtime)
        # the number of training steps in an epoch.
        stepNum = len(train_loader)
        numEpochs = args.epochs
        startTime = datetime.datetime.now()
        for epoch in range(startEpoch, numEpochs):
            # adjust learning rate
            #lr = stepLR.optimizer.param_groups[0]["lr"]
            lr = expLR.optimizer.param_groups[0]["lr"]
            
            print('\nEpoch: {} | LR: {:.16f}'.format(epoch + 1, lr))
            epochLosses = AverageMeter()
            for i, (features, labels) in enumerate(train_loader):  
                # Move tensors to the configured device
                features = features.reshape(-1, args.inputSize).to(device)
                labels = labels.reshape(-1, 2).to(device)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                epochLosses.update(loss.item(), features.size(0))
                #print(features.size(0))
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm(model.parameters(), 30)
                optimizer.step()

                # output the loss
                if (i+1) % int(stepNum/args.lossFrequency) == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, numEpochs, i+1, stepNum, loss.item()))
            
            # adjust learning rate
            #stepLR.step()
            expLR.step()
            endTime = datetime.datetime.now()
            totalTrainingTime = (endTime - startTime).seconds/60
            print('\nEpoch [{}/{}], Total Training Time: {:.2f} min'.format(epoch+1, numEpochs, totalTrainingTime))    

            # save the checkpoint
            if (epoch +1) % args.interval == 0:
                savePath = os.path.join(args.checkpoint, "checkpoint_epoch_{}.tar".format(str(epoch+1).zfill(3)))
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'lr': lr,
                 }, savePath)

        
        localtime = time.asctime(time.localtime(time.time()))
        print('\nTraining ends at ' + localtime)
        
    # test all the existing models.
    # load the existing models to test.
    if os.path.isdir(args.checkpoint):
        filelist = os.listdir(args.checkpoint)
        checkpoints = []
        checkpointNum = 0
        for name in filelist:
            # checkpoints are stored as tar files.
            if os.path.splitext(name)[-1][1:] == 'tar':
                checkpoints.append(name)
                checkpointNum +=1
        # test the checkpoints.
        if checkpointNum:
            print('\nCheckpoint Number : {}'.format(checkpointNum))
            checkpoints.sort()
            # load the test data.
            test_loader = LoadTestData(args.datasetDir, args.batchSize)
            # load the test labels.
            testY = np.load(args.datasetDir + 'testY.npy')
            testSize = testY.shape[0]
            # save the predictions.
            if args.savePrd:
                prdDir = args.prdDir
                RemakeDir(prdDir)
            localtime = time.asctime(time.localtime(time.time()))
            print('\nTest starts at ' + localtime)
            for name in checkpoints:
                print("\n==> Test checkpoint : {}".format(name))
                if device == torch.device('cuda'):
                    checkpoint = torch.load(args.checkpoint + name)
                    print('\nDevice: GPU')
                else:
                    checkpoint = torch.load(args.checkpoint + name, map_location=lambda storage, loc: storage)
                    print('\nDevice: CPU')
                            
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint['epoch']
                # the model's predictions.
                prdY = []
                # evaluate mode
                model.eval()
                epochLosses = AverageMeter()
                startTime = datetime.datetime.now()
                for i, (features, labels) in enumerate(test_loader): 
                    # Move tensors to the configured device
                    features = features.reshape(-1, args.inputSize).to(device)
                    labels = labels.reshape(-1, 2).to(device)
                    
                    # Forward pass
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    epochLosses.update(loss.item(), features.size(0))
                   
                    # save the outputs.
                    outputs_npy = outputs.data.cpu().detach().numpy()  
                    if(len(prdY) >0):
                        prdY = np.concatenate((prdY, outputs_npy))
                    else:
                        prdY = outputs_npy
                        
                endTime = datetime.datetime.now()
                # average predicting time for a single sample.
                avgTime = (endTime - startTime).seconds * 1000/testSize
                print('\nAverage prediction time: {:.3f} ms'.format(avgTime))
                
                # Calculate the prediction error (angular distance) between groundtruth and our predictions.
                prdError = np.zeros(testSize)
                for i in range(testSize):
                    prdError[i] = CalAngularDist(testY[i, 0:2], prdY[i, 0:2])
                meanPrdError = prdError.mean()
                sdPrdError = prdError.std()
                # standard error of the mean
                #SEM = prdError.std()/np.sqrt(testSize)
                print('Epoch: {}, Prediction Mean Error: {:.2f}, SD: {:.2f}'.format(epoch, meanPrdError, sdPrdError))
                
                # save the predictions.
                if args.savePrd:
                    prdDir = args.prdDir + 'predictions_epoch_{}/'.format(str(epoch).zfill(3))
                    MakeDir(prdDir)
                    predictions = np.zeros(shape = (testSize, 4))
                    predictions[:, 0:2] = testY
                    predictions[:, 2:4] = prdY
                    np.savetxt(prdDir + 'predictions.txt', predictions)
                  
            localtime = time.asctime(time.localtime(time.time()))
            print('\nTest ends at ' + localtime)   
        else:
            print('\n==> No valid checkpoints in directory {}'.format(args.checkpoint))
    else:
        print('\n==> Invalid checkpoint directory: {}'.format(args.checkpoint))
   
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description= 'FixationNet for DGazeDataset')
    
    # the number of input features
    parser.add_argument('--inputSize', default=1672, type=int,
                        help='the number of input features (default: 1672)')    
    # the size of gaze sequence data
    parser.add_argument('--gazeSeqSize', default=80, type=int,
                        help='the size of gaze sequence data (default: 80)')
    # the size of head sequence data
    parser.add_argument('--headSeqSize', default=80, type=int,
                        help='the size of head sequence data (default: 80)')    
    # the size of task sequence data
    parser.add_argument('--taskSeqSize', default=360, type=int,
                        help='the size of task sequence data (default: 360)')        
    # the size of saliency data
    parser.add_argument('--saliencySize', default=1152, type=int,
                        help='the size of saliency data (default: 1152)')    
    # the path of the cluster centers file
    parser.add_argument('--clusterPath', default='/data2/hzmData/Dataset/dataset/FixationNet_DGazeDataset_150_User1/clusterCenters.npy', type=str,
                        help='the path of the cluster centers file')
    # the directory that saves the dataset.
    parser.add_argument('-d', '--datasetDir', default = '/data2/hzmData/Dataset/dataset/FixationNet_DGazeDataset_150_User1/', type = str, 
                        help = 'the directory that saves the dataset')
    # trainFlag = 1 means train new models; trainFlag = 0 means test existing models.
    parser.add_argument('-t', '--trainFlag', default = 1, type = int, help = 'set the flag to train the model (default: 1)')
    # path to save checkpoint
    parser.add_argument('-c', '--checkpoint', default = '../checkpoint/FixationNet_DGazeDataset_150_User1/', type = str, 
                        help = 'path to save checkpoint')
    # save the prediction results or not.
    parser.add_argument('--savePrd', default = 0, type = int, help = 'save the prediction results (1) or not (0) (default: 0)')
    # the directory that saves the prediction results.
    parser.add_argument('-p', '--prdDir', default = '../predictions/FixationNet_DGazeDataset_150_User1/', type = str, 
                        help = 'the directory that saves the prediction results')
    # the number of total epochs to run
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of total epochs to run (default: 30)')
    # the batch size
    parser.add_argument('-b', '--batchSize', default=512, type=int,
                        help='the batch size (default: 512)')
    # the interval that we save the checkpoint
    parser.add_argument('-i', '--interval', default=5, type=int,
                        help='the interval that we save the checkpoint (default: 5)')
    # the initial learning rate.
    parser.add_argument('--lr', '--learningRate', default=1e-2, type=float,
                        help='initial learning rate (default: 1e-2)')
    parser.add_argument('--weightDecay', '--wd', default=5e-5, type=float,
                        help='weight decay (default: 5e-5)')
    parser.add_argument('--gamma', type=float, default=0.4,
                        help='Used to decay learning rate (default: 0.4)')
    # the loss function.
    parser.add_argument('--loss', default="AngularLoss", type=str,
                        help='Different loss to train the network: L1 | L2 | AngularLoss (default: AngularLoss)')
    # the frequency that we output the loss in an epoch.
    parser.add_argument('--lossFrequency', default=5, type=int,
                        help='the frequency that we output the loss in an epoch (default: 5)')
    main(parser.parse_args())
    