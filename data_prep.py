import numpy as np
import os
from scipy.ndimage import imread


def data_dict(bat_directory,sup_directory):

    #convert images to data points
    bat_dir = os.listdir(bat_directory)
    sup_dir = os.listdir(sup_directory)
    total_data={"images":[],"labels":[]}
    for i, j in zip(bat_dir, sup_dir):
        bat = imread(bat_directory+i, flatten=False)
        total_data["images"].append(bat)
        total_data["labels"].append("bat")
        sup = imread(sup_directory+j, flatten=False)
        total_data["images"].append(sup)
        total_data["labels"].append("sup")
    return total_data

def data_manipulator():
    global dic
    l = len(dic["images"])  # 202 images
    dic["images"] = (np.array(dic["images"])).reshape(l, -1)  # 202 images, 30*30*3=2700 features for each image
    dic["labels"] = (np.array(dic["labels"]))
    label_dic = {'labels': []}
    for i in dic['labels']:
        if i == 0:
            label_dic['labels'].append([0, 1])
        elif i == 1:
            label_dic['labels'].append([1, 0])
    label_dic["labels"] = (np.array(label_dic["labels"]))
    return dic['images'], label_dic['labels']

if __name__=="__main__":
    # load data
    bat_directory = "img_datasets/bat_img_dataset/"
    sup_directory = "img_datasets/sup_img_dataset/"

    #Convert as a dictionary
    dic = data_dict(bat_directory, sup_directory)

    #for convenience
    dic["images"] = np.array(dic["images"])
    for index,value in enumerate(dic["labels"]):
        if value=="bat":
            dic["labels"][index]=0
        if value=="sup":
            dic["labels"][index] = 1

    #finally, get the image_data and respective labels
    images,labels=data_manipulator()
