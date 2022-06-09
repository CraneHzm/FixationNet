## Solution Explanation

'FixationNet_Unity_Example' contains an example of running FixationNet model in Unity.  


"FixationNet_Server.py" loads a pre-trained FixationNet model and then waits for input data from Unity client to run the FixationNet model. Note that we only utilize gaze, head and task data as input features because saliency features are difficult to obtain in real time. The experimental results in our paper validate that using only gaze, head and task data can achieve good results.  


"Unity_Client.unity" collects gaze, head and task data and sends the data to the python server, i.e. "FixationNet_Server.py".  


"Unity_Client/Assets/Plugins/" contains the required netmq plugins.  


Unity Scripts:  
"CalculateHeadVelocity.cs": calculates the velocity of a head camera.    
"DataRecorder.cs": collects gaze, head and task data.   
"Client.cs": sends the collected data to a python server.   
"TrackObjects.cs": track the positions of the task-related objects in the scene.   


Using this example, you can do a lot of interesting things, e.g.  
1. Apply our pre-trained model to your Unity scene.  
2. Collect your own data to retrain our model or train your own model.  
3. Communicate between a Unity client and a Python server to do whatever you like.:)  


## Requirements:
Unity 2019.4.13+  
python 3.6+  
pytorch 1.1.0+  
pyzmq  
netmq  


## Usage:
Step 1: Run "FixationNet_Server/FixationNet_Server.py".  
Step 2: Use Unity to open "Unity_Client" and run "Unity_Client/Assets/Unity_Client.unity".  



