# FixationNet: Forecasting Eye Fixations in Task-Oriented Virtual Environments
Project homepage: https://cranehzm.github.io/FixationNet.


'FixationNet' contains the source code of our model and some pre-trained models.  


'FixationNet_Unity_Example' contains an example of running FixationNet model in Unity.


## Abstract
```
Human visual attention in immersive virtual reality (VR) is key for many important applications, such as content design, gaze-contingent rendering, or gaze-based interaction.
However, prior works typically focused on free-viewing conditions that have limited relevance for practical applications.
We first collect eye tracking data of 27 participants performing a visual search task in four immersive VR environments.
Based on this dataset, we provide a comprehensive analysis of the collected data and reveal correlations between users' eye fixations and other factors, i.e. users' historical gaze positions, task-related objects, saliency information of the VR content, and users' head rotation velocities.
Based on this analysis, we propose FixationNet -- a novel learning-based model to forecast users' eye fixations in the near future in VR.
We evaluate the performance of our model for free-viewing and task-oriented settings and show that it outperforms the state of the art by a large margin of 19.8% (from a mean error of 2.93 degrees to 2.35 degrees) in free-viewing and of 15.1% (from 2.05 degrees to 1.74 degrees) in task-oriented situations.
As such, our work provides new insights into task-oriented attention in virtual environments and guides future work on this important topic in VR research.
```	


## Environments:
Ubuntu: 18.04  
python 3.6+  
pytorch 1.1.0+  
cudatoolkit 10.0  
cudnn 7.6.5


## Usage:
Step 1: Download the dataset from our project homepage: https://cranehzm.github.io/FixationNet.

Step 2: Run the script "run_FixationNet.sh" in "FixationNet/scripts" directory to retrain or test our model on our dataset.
		Run the script "run_FixationNet_DGazeDataset.sh" in "FixationNet/scripts" directory to retrain or test our model on DGaze dataset.

