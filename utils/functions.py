import os
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torch.utils.data import DataLoader, Dataset
import torchvision
import torch.nn.functional as F
# function
def expToTorch(x):
    return torch.from_numpy(np.array(x)).type(torch.FloatTensor)
# 数据分类的label转换为tensor 这里要注意,一定要转换为longtensor类型
def labelTotorch(y):
    return torch.LongTensor(y)
def makeDataiter(x,y,batch_size,shuffle=True):
    return Data.DataLoader(Data.TensorDataset(x, y), batch_size, shuffle=shuffle)
def toLabel(ylabels):
    return torch.topk(ylabels, 1)[1].squeeze(1)
def toOneHot(ylabels,n_class):
    onehot = torch.zeros(ylabels.shape[0],n_class)
    index = torch.LongTensor(ylabels).view(-1,1)
    onehot.scatter_(dim=1, index=index, value=1)
    return onehot
def roc_plot(label=None,scores=None,title=None,file_name=None, save=False):
    fpr, tpr, threshold = roc_curve(label, scores)  ###计算真正率和假正率
    roc_auc = roc_auc_score(label, scores)  ###计算auc的值
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color=my_colors[3], lw=lw,
             label='AUC: %0.4f' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color=my_colors[0], lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('1 - Sepcificity', size=label_size)
    plt.ylabel('Sensitivity', size=label_size)
    plt.yticks(fontsize=ticks_size)
    plt.xticks(fontsize=ticks_size)
    plt.title(title, size=title_size)
    #plt.grid(True)
    plt.legend(bbox_to_anchor=legend_sit,
               fontsize=legend_size,
               borderaxespad=0.)
    if save:
        plt.savefig(file_name)
    plt.show()
    plt.clf()
# prepare model evaluation
def model_clf(model):
    model.train_epoch_loss = []
    model.train_epoch_acc = []
    model.train_epoch_aucroc = []
    model.val_epoch_loss = []
    model.val_epoch_acc = []
    model.val_epoch_aucroc = []
    model.test_predict = []
    model.test_sample_label = []
    model.light_encoder = []
    model.rgb_encoder = []
    return model
# data augment, only in train-set
def DataTransforms(phase=None):
    if phase == 'train':
        data = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(110),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.RandomRotation(45),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif phase == 'test':
        data = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(110),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return data
class MyDataset(Dataset):
    def __init__(self, DataFrame=None,transforms=None):
        
        super(MyDataset).__init__()
        self.rgb = DataFrame.image_path.values
        self.light = DataFrame.ligth_path.values
        self.label = DataFrame.label.values
        self.transforms = transforms
    def __len__(self):
        return len(self.rgb)
    def __getitem__(self, idx):
        rgb = Image.open(self.rgb[idx])
        light = Image.open(self.light[idx])
        label = self.label[idx] 
        rgb = self.transforms(rgb)
        light = self.transforms(light)
        return [rgb, light] ,label