# Copyright (c) Hu Zhiming 2022/06/08 jimmyhu@pku.edu.cn All Rights Reserved.

# run a pre-trained FixationNet model for a single input data.


from models.FixationNetModels import *
from utils import AngularCoord2ScreenCoord
import torch
import numpy as np
import zmq


# model parameters
gazeSeqSize = 80
headSeqSize = 80
taskSeqSize = 480
saliencySize = 1152
# saliency is not used as input features
inputSize = gazeSeqSize + headSeqSize + taskSeqSize

# path to the cluster centers file
clusterPath = './checkpoint/FixationNet_150_GazeHeadTask/clusterCenters.npy'
# path to a pre-trained model of predicting eye fixation in the future 150 ms.
modelPath = './checkpoint/FixationNet_150_GazeHeadTask/checkpoint_epoch_030.tar'


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the server
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")


def main():
	# Create the model
	print('\n==> Creating the model...')
	# We only utilize gaze, head and task data as input features because saliency features are difficult to obtain in real time. The experimental results in our paper validate that using only gaze, head and task data can achieve good results.	
	model = FixationNet_without_Saliency(gazeSeqSize, headSeqSize, taskSeqSize, saliencySize, clusterPath)
	model = torch.nn.DataParallel(model)
	if device == torch.device('cuda'):
		checkpoint = torch.load(modelPath)
		print('\nDevice: GPU')
	else:
		checkpoint = torch.load(modelPath, map_location=lambda storage, loc: storage)
		print('\nDevice: CPU')
	model.load_state_dict(checkpoint['model_state_dict'])                                          
	# evaluate mode
	model.eval()
	
	while True:
		#  Wait for next request from client
		message = socket.recv()		
		data = message.decode('utf-8').split(',')
		timeStamp = data[0]
		print("Time Stamp: {}".format(timeStamp))
		features = np.zeros((1, inputSize),  dtype=np.float32)
		for i in range(inputSize):			
			features[0, i] = float(data[i+1])
				
		singleInput = torch.tensor(features, dtype=torch.float32, device=device)			
		# Forward pass
		outputs = model(singleInput)
		outputs_npy = outputs.data.cpu().detach().numpy()[0]  			
		# The model outputs angular coordinates. Convert it to screen coordinates for better usage in Unity.
		# Angular coordinates: (0 deg, 0 deg) at screen center
		# Screen coordinates: (0, 0) at Bottom-left, (1, 1) at Top-right
		gaze = AngularCoord2ScreenCoord(outputs_npy)
		print("Eye Fixation in the Future 150 ms: {}".format(gaze))				
		gaze = str(gaze).encode('utf-8')
		socket.send(gaze)
		
if __name__ == '__main__':
    main()