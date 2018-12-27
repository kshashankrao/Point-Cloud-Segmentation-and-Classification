import numpy as np
import pcl
import os

p =  pcl.PointCloud()
x = np.array([[0,0,0]],dtype=np.float32)
path="D:/DeepLearning/PCL_Segmentation/point_net/pointnet/car_person_data/off/car/train"
save_path="D:/DeepLearning/PCL_Segmentation/point_net/pointnet/car_person_data/pcd/car/training/"

for root, dirs, files in os.walk(path):
    for filename in files:
        print("Processing: "+filename.split('.')[0])
        f = open(root+'/'+filename)
        lines = f.readlines()
        a = 0
        for line in lines:
            words = line.split()
            if (len(words) == 3) and float(words[0]) < 10000 :
                y = np.array([[words[0],words[1],words[2]]],dtype=np.float32)
                x = np.concatenate((x,y),axis=0)
        x = np.array(x,dtype=np.float32)
        print(x)
        point_cloud= pcl.load("test_cloud.pcd")
        point_cloud_np = point_cloud.to_array()
        point_cloud = p.from_array(x)
        print("Saved to path: "+save_path+filename.split('.')[0]+'.pcd')
        pcl.save(p,save_path+filename.split('.')[0]+'.pcd')
