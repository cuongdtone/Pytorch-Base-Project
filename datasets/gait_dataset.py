# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 11/10/2022


import os
import cv2
import torch
import glob
import random
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import sampler
import numpy as np


class GaitDataset(Dataset):
    def __init__(self, root):
        labels = os.listdir(root)  # Get list folder and file inside root
        self.classes_name = labels
        # print(labels)
        self.images = []
        self.labels = []
        for i in labels:
            path_l = os.path.join(root, i)
            images = os.listdir(path_l)
            abs_images_path = [os.path.join(path_l, img) for img in images]
            self.images += abs_images_path
            self.labels += [i] * len(abs_images_path)  # lay label

        # print(self.images)
        # print(self.labels)
        # pass
        # self.data = data
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)
        img = Image.fromarray(img)
        img = self.transform(img)
        # print(label)
        return img, self.classes_name.index(label)



if __name__ == '__main__':
    dataset = ImageFolderDataset(r'C:\Users\Cuong Tran\Desktop\DotLe\test_data')
    img, label = dataset.__getitem__(1)
    print(img.shape)
    # cv2.imshow("image", img)
    print(label)
    cv2.waitKey()