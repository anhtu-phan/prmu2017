import cv2
import os
import numpy as np
import csv

image_path = "../alcon2017prmu/dataset/characters"

number_to_filename = {}
nbimg_to_label = {}
label_to_unicode = {}
nb_label = 0
nb_images = 0
	
def build_data():
	global nb_label 	
	global nb_images

	#Buil dicts
	for file in os.listdir(image_path):
		absfile = os.path.join(image_path,file)
		##print(nb_label)
		label_to_unicode[nb_label] = file
		assert os.path.isdir(absfile)
		for file0 in os.listdir(absfile):
			file_name = os.path.join(absfile,file0)
			number_to_filename[nb_images] = file_name
			nbimg_to_label[nb_images] = nb_label
			nb_images = nb_images + 1
		
		nb_label = nb_label + 1
	#Write dicts to file
	with open('./map/dict_filename.csv','wb') as cvs_file :
		writer = csv.writer(cvs_file)
		for key,value in number_to_filename.items():
			writer.writerow([key, value])
	with open('./map/dic_img_label.csv','wb') as csv_file2 :
		writer2 = csv.writer(csv_file2)
		for key,value in nbimg_to_label.items():
			writer2.writerow([key,value])
	with open('./map/dic_label.csv','wb') as csv_file3 :
		writer3 = csv.writer(csv_file3)
		for key,value in label_to_unicode.items():
			writer3.writerow([key,value])

#def build_data_for_train(nb_images_train):



def read_data(arr_index_img):
	#batch_img = img_end - img_begin + 1
	X = np.zeros([len(arr_index_img),96,96,1])
	Y = np.zeros([len(arr_index_img),nb_label])

	index = 0
	for i in arr_index_img:
		file_name = number_to_filename[i]
		img = cv2.imread(file_name)
		img.resize(96,96,3)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		X[index,:,:,0] = img
		Y[index,nbimg_to_label[i]] = 1.0
		#print('img = ' + str(i) + '      img_label = '+str(nbimg_to_label[i]))
		index = index + 1
	##print Y
	return X,Y

#img = cv2.imread("/home/anhtu/characters/U+304A/U+304A_200021637-00005_2_X0349_Y2197.jpg")
#img = cv2.imshow('image',img)
#read_data()
#count_image()
##print(nb_label)
build_data()
