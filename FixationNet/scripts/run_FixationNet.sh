CUDA_VISIBLE_DEVICES="0" python FixationNet.py --trainFlag 0 --epochs 30 --lr 1e-2 --gamma 0.80 --batchSize 512 --loss AngularLoss --checkpoint ../checkpoint/FixationNet_150_User1/ --interval 5 --datasetDir ../../Dataset/dataset/FixationNet_150_User1/ --clusterPath /data2/hzmData/Dataset/dataset/FixationNet_150_User1/clusterCenters.npy --weightDecay 5e-5 