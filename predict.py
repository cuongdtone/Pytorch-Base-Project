# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 11/10/2022


import torch
import cv2
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image
from models.cnn import CNN

"""Load model"""
device = torch.device("cpu")
model = torch.load("src/cnn.pt", map_location=device)
model.to(device)
model.eval()
# print(model)


"""Load data"""


def pre_process(image):
    # img = cv2.imread(image)
    img = Image.fromarray(image)
    transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform(img)


img = cv2.imread(r'D:\Person Retrieval\Person Detect\samples\10\39.jpg')
x = pre_process(img)
x = x.unsqueeze(0)


"""Predict"""
y = model(x)
print(y.shape)
print(y)
_, pred = torch.max(y, 1)
print(pred)

percent = F.softmax(y, dim=1)
print(percent)


