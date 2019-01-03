# @Author: Shashank Rao
import os
import pcl
import h5py
import random
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
p =  pcl.PointCloud()
basepath = "D:/DeepLearning/PCL_Segmentation/point_net/pointnet/test_data"
#basepath = "D:/DeepLearning/PCL_Segmentation/point_net/pointnet/mesh_sampling/build/data/"

def rename(path):
	num = 0
	filenames = os.listdir(path)
	print(filenames)
	for filename in filenames:
		if filename != "helper_project.py":
			os.rename((filename),"car_"+str(num)+".pcd" )
			num+=1
			print(num)

def list_files(basePath, validExts=None, contains=None):
	for (rootDir, dirNames, filenames) in os.walk(basePath):
		for filename in filenames:
			if contains is not None and filename.find(contains) == -1:
				continue
			ext = filename[filename.rfind("."):].lower()
			if validExts is None or ext.endswith(validExts):
				imagePath = os.path.join(rootDir, filename)
				yield imagePath

def convert2h5():
		data=[]
		labels = []
		corrected = []
		pcdPaths = sorted((list(list_files(basepath,validExts=".pcd",contains=None))))
		random.seed(42)
		random.shuffle(pcdPaths)
		for pcdPath in pcdPaths:
			point_cloud= pcl.load(pcdPath)
			pcdPath = os.path.abspath(pcdPath)
			point_cloud_np = point_cloud.to_array()
			print("Path: "+pcdPath)
			print("Original Shape: "+str(point_cloud_np.shape))
			if (point_cloud_np.shape[0] < 2048):
				offset = 2048 - point_cloud_np.shape[0]
				corrected.append(pcdPath)
				for i in range(offset):
					point_cloud_np = np.vstack([point_cloud_np,point_cloud_np[-1]])
				print("Corrected Shape: "+str(point_cloud_np.shape))
			points = point_cloud_np
			#centroid = np.mean(points, axis=0)
			#print(centroid)
			#points -= centroid
			furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
			print(furthest_distance)
			points /= furthest_distance
			print(points)
			data.append(points)
			labelName = pcdPath.split("\\")[-1].split("_")[-2]
			print(labelName+"\n")
			if(labelName == "car"):
				label = 0
			else:
				label = 1
			# Never Do this label = 1
			print("Class: "+labelName+"\n"+"Label: "+str(label)+"\n")
			labels.append(label)
		labelss = np.asarray(labels)
		labelss.shape = [labelss.shape[0],1]
		#print(labelss)
		#split = train_test_split(data, labelss,test_size=0.2, random_state=42)
		#trainX, trainY, labelX, labelY = split
		data = np.asarray(data)
		#print(data.shape)
		print("Corrected Information: \n"+str(len(corrected)))
		hf_train = h5py.File('test.h5','w')
		hf_train.create_dataset('data',data=data)
		hf_train.create_dataset('label',data=labelss)
		hf_train.close()
		#print(labelss.shape)
		# hf_val = h5py.File('test_data.h5','w')
		# hf_val.create_dataset('data',data=trainY)
		# hf_val.create_dataset('label',data=labelY)
		# hf_val.close()

def loadh5(filename):
	f = h5py.File(filename)
	data = f['data'][:]
	label = f['label'][:]
	print (label)
	return data, label

def h5toarray(filename):
	hf = h5py.File(filename, 'r')
	data = hf.get('data').value
	print(data)
	#p =  pcl.PointCloud()
	pcd_new = p.from_array(data[1])
	pcl.save(p,"test.pcd")

def augment_data(filename,rotation_angle):
	batch_data, label = loadh5(filename)
	rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
	for k in range(batch_data.shape[0]):
		cosval = np.cos(rotation_angle)
		sinval = np.sin(rotation_angle)
		rotation_matrix = np.array([[cosval, 0, sinval],
									[0, 1, 0],
									[-sinval, 0, cosval]])
		shape_pc = batch_data[k,:,0:3]
		rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
	hf_train = h5py.File('train_rotated_90.h5','w')
	hf_train.create_dataset('data',data=rotated_data)
	hf_train.create_dataset('label',data=label)
	hf_train.close()

def extractDatafromH5(filename):
	test = 0
	f = h5py.File(filename)
	data = f['data'][:]
	label = f['label'][:]
	print (label.shape[0])
	for i in range(label.shape[0]):
		if label[i] == 0:
			print(data[i])
			pcd_new = p.from_array(data[i])
			pcl.save(p,"del/car/car_"+str(test+110)+".pcd")
			test+=1
	print(test)

def scale_point_cloud(filename, scale_low=0.8, scale_high=1.25):
	batch_data, label = loadh5(filename)
	B, N, C = batch_data.shape
	scales = np.random.uniform(scale_low, scale_high, B)
	for batch_index in range(B):
		batch_data[batch_index,:,:] *= scales[batch_index]
	hf_train = h5py.File('train_scaled.h5','w')
	hf_train.create_dataset('data',data=batch_data)
	hf_train.create_dataset('label',data=label)
	hf_train.close()

#rename("D:/DeepLearning/PCL_Segmentation/point_net/pointnet/mesh_sampling/build/data/car")
#convert2h5()
#loadh5("train_scaled.h5")
#5toarray("Data_test.h5")
extractDatafromH5("train_scaled.h5")
#augment_data("train.h5",90)
#scale_point_cloud("train.h5")
