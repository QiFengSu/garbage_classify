#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：garbage_classify 
@File    ：resnet101.py
@IDE     ：PyCharm 
@Author  ：AC_sqf
@Date    ：2022/5/28 17:07
'''
import logging
import torch
import torch.nn as nn
import os

# import tqdm as tqdm
from tqdm import tqdm
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
# img_name = os.path.join('./garbage_classify_v2/train_data_v2','img_1.jpg')
# image = Image.open(img_name)
# image.show()
import pandas as pd

examples = pd.read_csv('datas.csv')
# set log format
logging.basicConfig(level=logging.INFO, filename='./log.txt', filemode='w',
                    format='%(asctime)-15s %(levelname)s: %(message)s')


class ImageDataset(Dataset):
    def __init__(self, sample, transform):
        self.data = sample  # 存储的是图片的名称
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('./garbage_classify_v2/train_data_v2', self.data.loc[idx, 'data'])
        label = self.data.loc[idx, 'label']
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': label}


data_transform = transforms.Compose([
    # transforms.Resize(224),
    # transforms.RandomCrop(224),
    # transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transforms.Normalize(mean=[0.47145638718882443, 0.5182074217491532, 0.5559518648087405],std=[0.2878283895184953, 0.2726498177701412, 0.27027624285255336])
    # transforms.Normalize([0.3986, 0.2548, 0.1389], [0.3013, 0.2124, 0.1443])
])

data_transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    ##transforms.RandomHorizontalFlip(p=0.5),
    # transforms.Grayscale(num_output_channels=3),
    # transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transforms.Normalize(mean=[0.47145638718882443, 0.5182074217491532, 0.5559518648087405],std=[0.2878283895184953, 0.2726498177701412, 0.27027624285255336])
    # transforms.Normalize([0.3986, 0.2548, 0.1389], [0.3013, 0.2124, 0.1443])
])
# from sklearn.model_selection import StratifiedShuffleSplit
# split = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
# for train_index, test_index in split.split(sample, sample["label"]):
#     train_dataset = sample.loc[train_index].reset_index().drop('index',axis=1)
#     test_dataset = sample.loc[test_index].reset_index().drop('index',axis=1)
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(examples, examples['label']):
    train_dataset = examples.loc[train_idx].reset_index().drop('index', axis=1)
    test_dataset = examples.loc[test_idx].reset_index().drop('index', axis=1)

train_dataSet = ImageDataset(train_dataset, data_transform)
train_dataloader = DataLoader(train_dataSet, batch_size=16, shuffle=True)

test_dataset = ImageDataset(test_dataset, data_transform_test)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
print(next(iter(train_dataloader)))

# 获取模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
# model.fc = nn.Linear(2048, 40)

model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(2048, 40)
    )

# 定义学习率等
# optimizer = optim.Adam(model.fc.parameters(), lr=0.005)  #,weight_decay=0.0001
optimizer = optim.SGD(model.fc.parameters(), lr=0.005, momentum=0.9, weight_decay=0.001)
# optimizer = optim.RMSprop(model._fc.parameters(), lr=0.01, momentum=0.9,eps=0.001, weight_decay=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
criterion = nn.CrossEntropyLoss()
# criterion = LabelSmoothingLoss(54,0.1)
best_acc = 0
best_epoch = 0
num_epochs = 20
batch_size = 16
checkpoint_interval = 3
start_epoch = 0

# path_checkpoint = "./checkpoint_4_epoch.pkl"#断点路径
# 断点续传

path_checkpoint = ''
if path_checkpoint != '':
    checkpoint = torch.load(path_checkpoint, map_location=device)#加载断点
    model.load_state_dict(checkpoint['model_state_dict'])#加载模型可学习参数
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])#加载优化器参数
    start_epoch = checkpoint['epoch']#设置开始的epoch
    scheduler.last_epoch = start_epoch#设置学习率的last_epoch

ct = 0
# 这里的目的是防止预训练模型的准确结果随训练进行变化。
for child in model.children():
    ct += 1
    # print(ct,child)
    if ct < 5:
        for param in child.parameters():
            param.requires_grad = False
model = model.to(device)

for epoch in range(start_epoch, num_epochs):
    logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
    logging.info('-' * 10)
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    model.train()
    running_loss = 0.0
    tk0 = tqdm(train_dataloader, total=int(len(train_dataloader)))
    counter = 0
    labellist = []
    pred = []
    correct = 0
    scheduler.step()
    for i, x_batch in enumerate(tk0):
        inputs = x_batch['image']
        labels = x_batch['label'].view(-1).long()
        # ones = torch.sparse.torch.eye(2)
        # labels = ones.index_select(0,labels)
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        optimizer.zero_grad()
        # sm = torch.nn.Softmax(dim = 1)

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        counter += 1
        tk0.set_postfix(loss=(running_loss / (counter * train_dataloader.batch_size)))
        for flag in x_batch['label'].view(-1):
            labellist.append(flag.item())
        sm = torch.nn.Softmax(dim=1)
        pred = sm(outputs).data.max(1, keepdim=True)[1]
        # print(labels.view_as(pred).cpu())
        correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
    # print(correct)

    epoch_loss = running_loss / len(train_dataloader)
    logging.info('Train acc: {:.6f}'.format(correct / len(train_dataloader) / batch_size))
    logging.info('Training Loss: {:.4f}'.format(epoch_loss))
    print('Train acc:', correct / len(train_dataloader) / batch_size)
    print('Training Loss: {:.4f}'.format(epoch_loss))

    # 存储断点，每训练 3 次存储一次
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch}
        path_checkpoint = "./checkpointfirst_{}_epoch.pkl".format(epoch)
        torch.save(checkpoint, path_checkpoint)

    tk1 = tqdm(test_dataloader, total=int(len(test_dataloader)))
    labels = []
    pred = []
    correct = 0
    model.eval()
    for i, x_batch in enumerate(tk1):
        inputs = x_batch['image']
        labels = x_batch['label'].view(-1).to(device, dtype=torch.long)
        # print(labels)
        with torch.no_grad():
            sm = torch.nn.Softmax(dim=1)
            pred = sm(model(inputs.to(device, dtype=torch.float))).data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).cpu().sum().item()

    logging.info('Test acc: {:.6f}'.format(correct / len(test_dataloader) / batch_size))
    print('Test acc:', correct / len(test_dataloader) / batch_size)
# if correct/len(test_dataloader)/batch_size>best_acc:
# torch.save(model.state_dict(), "model_resnext_best.pth")
# best_acc=correct/len(test_dataloader)/batch_size
# best_epoch=epoch
# print(best_acc,best_epoch)
torch.save(model.state_dict(),
           "model_resnext_firsttune_gem" + str(correct / len(test_dataloader) / batch_size) + "_1128.pth")

# 定义学习率等
# optimizer = optim.Adam(model.fc.parameters(), lr=0.005)  #,weight_decay=0.0001
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.001)
# optimizer = optim.RMSprop(model._fc.parameters(), lr=0.01, momentum=0.9,eps=0.001, weight_decay=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
criterion = nn.CrossEntropyLoss()
# criterion = LabelSmoothingLoss(54,0.1)

# path_checkpoint = "./checkpoint_4_epoch.pkl"#断点路径
# 断点续传
path_checkpoint = ''
if path_checkpoint != '':
    checkpoint = torch.load(path_checkpoint)#加载断点
    model.load_state_dict(checkpoint['model_state_dict'])#加载模型可学习参数
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])#加载优化器参数
    start_epoch = checkpoint['epoch']#设置开始的epoch
    scheduler.last_epoch = start_epoch#设置学习率的last_epoch



best_acc = 0
best_epoch = 0
num_epochs = 20
batch_size = 16
for epoch in range(num_epochs):
    logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
    logging.info('-' * 10)
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    model.train()
    running_loss = 0.0
    tk0 = tqdm(train_dataloader, total=int(len(train_dataloader)))
    counter = 0
    labellist = []
    pred = []
    correct = 0
    scheduler.step()
    for i, x_batch in enumerate(tk0):
        inputs = x_batch['image']
        labels = x_batch['label'].view(-1).long()
        # ones = torch.sparse.torch.eye(2)
        # labels = ones.index_select(0,labels)
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        optimizer.zero_grad()
        # sm = torch.nn.Softmax(dim = 1)

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        counter += 1
        tk0.set_postfix(loss=(running_loss / (counter * train_dataloader.batch_size)))
        for flag in x_batch['label'].view(-1):
            labellist.append(flag.item())
        sm = torch.nn.Softmax(dim=1)
        pred = sm(outputs).data.max(1, keepdim=True)[1]
        # print(labels.view_as(pred).cpu())
        correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
    # print(correct)

    epoch_loss = running_loss / len(train_dataloader)
    logging.info('Train acc: {:.6f}'.format(correct / len(train_dataloader) / batch_size))
    logging.info('Training Loss: {:.4f}'.format(epoch_loss))
    print('Train acc:', correct / len(train_dataloader) / batch_size)
    print('Training Loss: {:.4f}'.format(epoch_loss))

    # 存储断点，每训练 3 次存储一次
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch}
        path_checkpoint = "./checkpointsecond_{}_epoch.pkl".format(epoch)
        torch.save(checkpoint, path_checkpoint)

    tk1 = tqdm(test_dataloader, total=int(len(test_dataloader)))
    labels = []
    pred = []
    correct = 0
    model.eval()
    for i, x_batch in enumerate(tk1):
        inputs = x_batch['image']
        labels = x_batch['label'].view(-1).to(device, dtype=torch.long)
        # print(labels)
        with torch.no_grad():
            sm = torch.nn.Softmax(dim=1)
            pred = sm(model(inputs.to(device, dtype=torch.float))).data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
    logging.info('Test acc: {:.6f}'.format(correct / len(test_dataloader) / batch_size))
    print('Test acc:', correct / len(test_dataloader) / batch_size)
# if correct/len(test_dataloader)/batch_size>best_acc:
# torch.save(model.state_dict(), "model_resnext_best.pth")
# best_acc=correct/len(test_dataloader)/batch_size
# best_epoch=epoch
# print(best_acc,best_epoch)
torch.save(model.state_dict(),
           "model_resnext_secondtune_gem" + str(correct / len(test_dataloader) / batch_size) + "_1128.pth")
