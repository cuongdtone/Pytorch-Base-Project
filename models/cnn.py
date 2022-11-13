# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 11/10/2022

import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self):  # dinh nghia cau truc mo hinh: 2 lop tich chap, 2 lop ket noi day du
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()  # chuyen feature map thanh 1 vecto
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1605632, out_features=84),  # lop ket noi day du
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=2)
        )

    def forward(self, x):
        feature = self.feature(x)
        out = self.classifier(feature)
        return out


if __name__ == '__main__':
    model = CNN()
    x = torch.rand(4, 3, 112, 112)
    y = model(x)
    print(y)

