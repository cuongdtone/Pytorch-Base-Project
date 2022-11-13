# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 11/10/2022


import torch
import cv2
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image
from models.ae import AE
import numpy as np
import matplotlib.pyplot as plt


"""Load model"""
device = torch.device("cpu")
model = AE()
model.load_state_dict(torch.load("src/ae_epoch_19_loss_0.04.pt"))
model.to(device)
model.eval()
# print(model)


"""Load data"""


def crop(mask):
    x, y, w, h = cv2.boundingRect(mask)
    crop = mask[y:y + h, x:x + w]
    if h > w:
        bg = np.zeros((h, h))
        st_x = int(h / 2 - w / 2)
        ed_x = int(h / 2 + w / 2)
        bg[:, st_x:ed_x] = crop
    else:
        bg = np.zeros((w, w))
        st_y = int(w / 2 - h / 2)
        ed_y = int(w / 2 + h / 2)
        bg[st_y:ed_y, :] = crop
    return bg


def pre_process(image):
    # img = cv2.imread(image)
    image = crop(image)
    img = Image.fromarray(image)
    transform_list = [transforms.Grayscale(1),
                      transforms.Resize(112),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5,), (0.5,))]
    transform = transforms.Compose(transform_list)
    return transform(img)


img = cv2.imread(r'D:\dataset\001\bg-01_000\001-bg-01-000-001.png', cv2.IMREAD_GRAYSCALE)
# print(img.shape)
x = pre_process(img)
x = x.unsqueeze(0)


"""Predict"""
# y = model(x)
# plt.imshow(y[0][0].detach().numpy())
# plt.show()
# print(y.shape)
# print(y)

"""Code extractor"""
embed = model.encoder(x)
print(embed.shape)
embed = embed.view(-1)
print(embed.shape)

####
# y = model.decoder(embed)
# plt.imshow(y[0][0].detach().numpy())
# plt.show()
# print(y.shape)
# print(y)



