import h5py
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import csv
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from os import listdir
from os.path import isfile, join

# train_data = (pd.read_hdf('./train/train_data_processed.h5','table')).values
# img = load_img('./train/'+train_data[0][1])
# print(img)
# crop = img.crop( (train_data[0][3], train_data[0][4],train_data[0][7],train_data[0][6]) )
# crop.show()

def load_img(path):
    return Image.open(path).convert('L')

class Load_traindata(Dataset):
    def __init__(self, transform=None, target_transform=None, loader=load_img, valid=False, valid_len=1000):
        # initial variable
        self.imgs = []
        train_data = (pd.read_hdf('./train/train_data_processed.h5','table')).values
        for i in range(len(train_data)):
            # print(train_data[i])
            self.imgs.append(train_data[i])
        self.transform = transform
        self.target_transform = target_transform
        self.loader = load_img
        
    def __getitem__(self, index):
        filename, label = self.imgs[index][1], self.imgs[index][2]
        left, top, right, bottom = self.imgs[index][3], self.imgs[index][4], self.imgs[index][7], self.imgs[index][6]
        img = self.loader('./train/'+filename)
        crop_img = img.crop((left, top, right, bottom))
        if self.transform is not None:
            crop_img = self.transform(crop_img)
        return crop_img, label

    def __len__(self):
        return len(self.imgs)

# Load_traindata()