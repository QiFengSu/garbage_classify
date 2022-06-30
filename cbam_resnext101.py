#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：garbage_classify 
@File    ：cbam_resnext101.py
@IDE     ：PyCharm 
@Author  ：AC_sqf
@Date    ：2022/6/30 15:00 
'''
import logging
import torch
import torch.nn as nn
import os

# import tqdm as tqdm
from torch.hub import load_state_dict_from_url
from tqdm import tqdm
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision.models.resnet import Bottleneck
from cbam import ResNet
# img_name = os.path.join('./garbage_classify_v2/train_data_v2','img_1.jpg')
# image = Image.open(img_name)
# image.show()
import pandas as pd

examples = pd.read_csv('/content/drive/MyDrive/garbage_classify/datas.csv')
# set log format
logging.basicConfig(level=logging.INFO, filename='/content/drive/MyDrive/garbage_classify/log.txt', filemode='w',
                    format='%(asctime)-15s %(levelname)s: %(message)s')

def _resnext(arch, block, layers, pretrained, progress, **kwargs):
    # 自己重写的网络
    model = ResNet(block, layers, **kwargs)
    model.fc = nn.Linear(2048, 40)
    print(1)
    pretrained_dict = load_state_dict_from_url('https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth', progress=progress)

    model_dict = model.state_dict()  # 网络层的参数
    # 需要加载的预训练参数
    # pretrained_dict = torch.load(model_path)['state_dict']  # torch.load得到是字典，我们需要的是state_dict下的参数
    pretrained_dict = {k.replace('module.', ''): v for k, v in
                       pretrained_dict.items()}  # 因为pretrained_dict得到module.conv1.weight，但是自己建的model无module，只是conv1.weight，所以改写下。

    # 删除pretrained_dict.items()中model所没有的东西
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 只保留预训练模型中，自己建的model有的参数
    model_dict.update(pretrained_dict)  # 将预训练的值，更新到自己模型的dict中
    model.load_state_dict(model_dict)  # model加载dict中的数据，更新网络的初始值

    # model.load_state_dict(state_dict)
    # model.load_state_dict(torch.load('../model_resnext_firsttune_gem0.9137228260869565_1128.pth', map_location=device))
    ct = 0
    # 这里的目的是防止预训练模型的准确结果随训练进行变化。
    for child in model.children():
        ct += 1
        # print(ct,child)
        if ct == 4 or ct == 5:
            continue
        for param in child.parameters():
            param.requires_grad = False
    model = model.to(device)
    print(2)
    return model

def resnext101_32x16d_wsl(progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 16
    return _resnext('resnext101_32x16d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)

class ImageDataset(Dataset):
    def __init__(self, sample, transform):
        self.data = sample  # 存储的是图片的名称
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('/content/drive/MyDrive/garbage_classify/garbage_classify_v2/train_data_v2', self.data.loc[idx, 'data'])
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

batch_size_set = 24
split = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(examples, examples['label']):
    train_dataset = examples.loc[train_idx].reset_index().drop('index', axis=1)
    test_dataset = examples.loc[test_idx].reset_index().drop('index', axis=1)

train_dataSet = ImageDataset(train_dataset, data_transform)
train_dataloader = DataLoader(train_dataSet, batch_size=batch_size_set, shuffle=True)

test_dataset = ImageDataset(test_dataset, data_transform_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size_set, shuffle=True)

# 获取模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
model.fc = nn.Linear(2048, 40)


# 定义学习率等
# optimizer = optim.Adam(model.fc.parameters(), lr=0.005)  #,weight_decay=0.0001
# optimizer = optim.SGD(model.fc.parameters(), lr=0.005, momentum=0.9, weight_decay=0.001)
# optimizer = optim.RMSprop(model._fc.parameters(), lr=0.01, momentum=0.9,eps=0.001, weight_decay=0.001)
optimizer = optim.SGD(model.fc.parameters(), lr=0.005, momentum=0.9, weight_decay=0.001)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.CrossEntropyLoss()
# criterion = LabelSmoothingLoss(54,0.1)
best_acc = 0
best_epoch = 0
num_epochs = 40
batch_size = 24
checkpoint_interval = 3
start_epoch = 0

# path_checkpoint = "/content/drive/MyDrive/garbage_classify/checkpointfirst_17_epoch.pkl"#断点路径
# path_checkpoint = "/content/drive/MyDrive/garbage_classify/model_resnext_firsttune_gem0.9137228260869565_1128.pth"
# 断点续传


# checkpoint = torch.load(path_checkpoint)#加载断点
# model.load_state_dict(checkpoint)#加载模型可学习参数
# model = model.to(device)


# start_epoch = 20  #设置开始的epoch
scheduler.last_epoch = start_epoch#设置学习率的last_epoch

# ct = 0
# # 这里的目的是防止预训练模型的准确结果随训练进行变化。
# for child in model.children():
#     ct += 1
#     # print(ct,child)
#     if ct < 5:
#         for param in child.parameters():
#             param.requires_grad = False
# model = model.to(device)

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
        path_checkpoint = "/content/drive/MyDrive/garbage_classify/checkpointfirst_{}_epoch.pkl".format(epoch)
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
           "/content/drive/MyDrive/garbage_classify/model_resnext_firsttune2_gem" + str(correct / len(test_dataloader) / batch_size) + "_1128.pth")

# # 定义学习率等
# # optimizer = optim.Adam(model.fc.parameters(), lr=0.005)  #,weight_decay=0.0001
# optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.001)
# # optimizer = optim.RMSprop(model._fc.parameters(), lr=0.01, momentum=0.9,eps=0.001, weight_decay=0.001)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
# criterion = nn.CrossEntropyLoss()
# # criterion = LabelSmoothingLoss(54,0.1)

# # path_checkpoint = "./checkpoint_4_epoch.pkl"#断点路径
# # 断点续传
# path_checkpoint = ''
# if path_checkpoint != '':
#     checkpoint = torch.load(path_checkpoint)#加载断点
#     model.load_state_dict(checkpoint['model_state_dict'])#加载模型可学习参数
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])#加载优化器参数
#     start_epoch = checkpoint['epoch']#设置开始的epoch
#     scheduler.last_epoch = start_epoch#设置学习率的last_epoch



# best_acc = 0
# best_epoch = 0
# num_epochs = 20
# batch_size = 16
# for epoch in range(num_epochs):
#     logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
#     logging.info('-' * 10)
#     print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#     print('-' * 10)

#     model.train()
#     running_loss = 0.0
#     tk0 = tqdm(train_dataloader, total=int(len(train_dataloader)))
#     counter = 0
#     labellist = []
#     pred = []
#     correct = 0
#     scheduler.step()
#     for i, x_batch in enumerate(tk0):
#         inputs = x_batch['image']
#         labels = x_batch['label'].view(-1).long()
#         # ones = torch.sparse.torch.eye(2)
#         # labels = ones.index_select(0,labels)
#         inputs = inputs.to(device, dtype=torch.float)
#         labels = labels.to(device, dtype=torch.long)
#         optimizer.zero_grad()
#         # sm = torch.nn.Softmax(dim = 1)

#         with torch.set_grad_enabled(True):
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#         running_loss += loss.item() * inputs.size(0)
#         counter += 1
#         tk0.set_postfix(loss=(running_loss / (counter * train_dataloader.batch_size)))
#         for flag in x_batch['label'].view(-1):
#             labellist.append(flag.item())
#         sm = torch.nn.Softmax(dim=1)
#         pred = sm(outputs).data.max(1, keepdim=True)[1]
#         # print(labels.view_as(pred).cpu())
#         correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
#     # print(correct)

#     epoch_loss = running_loss / len(train_dataloader)
#     logging.info('Train acc: {:.6f}'.format(correct / len(train_dataloader) / batch_size))
#     logging.info('Training Loss: {:.4f}'.format(epoch_loss))
#     print('Train acc:', correct / len(train_dataloader) / batch_size)
#     print('Training Loss: {:.4f}'.format(epoch_loss))

#     # 存储断点，每训练 3 次存储一次
#     if (epoch + 1) % checkpoint_interval == 0:
#         checkpoint = {"model_state_dict": model.state_dict(),
#                       "optimizer_state_dict": optimizer.state_dict(),
#                       "epoch": epoch}
#         path_checkpoint = "/content/drive/MyDrive/garbage_classify/checkpointsecond_{}_epoch.pkl".format(epoch)
#         torch.save(checkpoint, path_checkpoint)

#     tk1 = tqdm(test_dataloader, total=int(len(test_dataloader)))
#     labels = []
#     pred = []
#     correct = 0
#     model.eval()
#     for i, x_batch in enumerate(tk1):
#         inputs = x_batch['image']
#         labels = x_batch['label'].view(-1).to(device, dtype=torch.long)
#         # print(labels)
#         with torch.no_grad():
#             sm = torch.nn.Softmax(dim=1)
#             pred = sm(model(inputs.to(device, dtype=torch.float))).data.max(1, keepdim=True)[1]
#             correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
#     logging.info('Test acc: {:.6f}'.format(correct / len(test_dataloader) / batch_size))
#     print('Test acc:', correct / len(test_dataloader) / batch_size)
# # if correct/len(test_dataloader)/batch_size>best_acc:
# # torch.save(model.state_dict(), "model_resnext_best.pth")
# # best_acc=correct/len(test_dataloader)/batch_size
# # best_epoch=epoch
# # print(best_acc,best_epoch)
# torch.save(model.state_dict(),
#            "/content/drive/MyDrive/garbage_classify/model_resnext_secondtune_gem" + str(correct / len(test_dataloader) / batch_size) + "_1128.pth")
