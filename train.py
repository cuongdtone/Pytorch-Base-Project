# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 11/10/2022

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from datasets.dataset import ImageFolderDataset
from models.cnn import CNN

""" Dinh nghia sieu tham so"""
epochs = 2
batch_size = 8
lr = 0.001
data_root = "test_data"

"""Load data"""
dataset = ImageFolderDataset(data_root)
data_loader = DataLoader(dataset, batch_size, shuffle=True)

"""Dinh nghia model"""
device = torch.device("cpu")
model = CNN()
model.to(device=device)
model.train()

"""Dinh nghia ham mat mat va toi uu"""
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)

"""train model"""
for epoch in range(epochs):
    model.train()
    accs = []
    losses = []
    for b, (imgs, labels) in enumerate(data_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        # forward
        out_model = model(imgs)
        # loss calc
        loss = criterion(out_model, labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(out_model, 1)
        acc = torch.sum(preds == labels.data)/len(imgs)
        losses.append(loss.item())
        accs.append(acc)
        print(f"Epoch {epoch}, {b}/{len(dataset)/batch_size}, loss={loss.item()}, acc = {acc}")
    print(f'Ket qua epoch {epoch}')
    print(sum(losses)/len(losses))
    print(sum(accs)/len(accs))

    """eval"""
    model.eval()


torch.save(model, 'src/cnn.pt')

