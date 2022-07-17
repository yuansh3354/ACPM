import os
import cv2
import yaml
import torch
import warnings
import numpy as np
import torchvision
import pandas as pd
import torch.nn as nn
from PIL import Image
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import Counter
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
import torchmetrics
from sklearn.model_selection import train_test_split
# my imports
from utils.functions import *

class Mish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * (torch.tanh(F.softplus(input)))

# 初始化基本densen参数
class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', Mish())
        self.add_module(
            'conv1',
            nn.Conv2d(in_channels,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', Mish())
        self.add_module(
            'conv2',
            nn.Conv2d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))

    # 重载forward函数
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module(
                'denselayer%d' % (i + 1),
                _DenseLayer(in_channels + growth_rate * i, growth_rate,
                            bn_size))


class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', Mish())
        self.add_module(
            'conv',
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


def bulid_layers(layers, block_config, layers_name, num_feature, bn_size,
                 growth_rate, theta):
    for i, num_layers in enumerate(block_config):
        layers.add_module(
            layers_name + 'denseblock%d' % (i + 1),
            _DenseBlock(num_layers, num_feature, bn_size, growth_rate))

        num_feature = num_feature + growth_rate * num_layers
        if i != len(block_config) - 1:
            layers.add_module(
                layers_name + 'transition%d' % (i + 1),
                _Transition(num_feature, int(num_feature * theta)))
            num_feature = int(num_feature * theta)
    return layers,num_feature


# 构建双路DenseNet
class Double_Branch_Dense_Net(nn.Module):
    def __init__(self,
                 growth_rate=12,
                 block_config=[(6,12,4), (6,12,4)],
                 bn_size=4,
                 theta=0.5,
                 num_classes=2):
        super(Double_Branch_Dense_Net, self).__init__()

        # 定义双路网络
        # 亮度图网络
        l_num_init_feature = 2 * growth_rate
        self.light_features = nn.Sequential(
            OrderedDict([('conv0',
                          nn.Conv2d(3,
                                    l_num_init_feature,
                                    kernel_size=7,
                                    stride=2,
                                    padding=3,
                                    bias=False)),
                         ('norm0', nn.BatchNorm2d(l_num_init_feature)),
                         ('relu0', Mish()),
                         ('pool0',
                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        l_num_feature = l_num_init_feature
        self.light_features,l_num_feature = bulid_layers(self.light_features,
                                           block_config[0], 'light_',
                                           l_num_feature, bn_size, growth_rate,
                                           theta)
        self.light_features.add_module('norm5', nn.BatchNorm2d(l_num_feature))
        self.light_features.add_module('relu5', Mish())
        self.light_features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
        
        
        #　rgb网络
        r_num_init_feature = 2 * growth_rate
        self.rgb_features = nn.Sequential(
            OrderedDict([('rgb_conv0',
                          nn.Conv2d(3,
                                    r_num_init_feature,
                                    kernel_size=7,
                                    stride=2,
                                    padding=3,
                                    bias=False)),
                         ('rgb_norm0', nn.BatchNorm2d(r_num_init_feature)),
                         ('rgb_relu0', Mish()),
                         ('rgb_pool0',
                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        r_num_feature = r_num_init_feature
        self.rgb_features,r_num_feature = bulid_layers(self.rgb_features, block_config[1],
                                         'rgb_', r_num_feature, bn_size,
                                         growth_rate, theta)
        self.rgb_features.add_module('norm5', nn.BatchNorm2d(r_num_feature))
        self.rgb_features.add_module('relu5', Mish())
        self.rgb_features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))

        self.classifier = nn.Linear(l_num_feature + r_num_feature,
                                    num_classes,
                                    bias=False)

    def forward(self, inputs):
        l_features = self.light_features(inputs[0])
        r_features = self.rgb_features(inputs[1])
        features = torch.cat([l_features, r_features], 1)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        l = l_features.cpu().detach().numpy()
        r = r_features.cpu().detach().numpy()
        return out,l,r

class DensNet(pl.LightningModule):
    # 用于存放训练输出数据
    train_epoch_loss = []
    train_epoch_acc = []
    train_epoch_aucroc = []
    # 用于存放验证集输出
    val_epoch_loss = []
    val_epoch_acc = []
    val_epoch_aucroc = []
    # 用于存放测试集输出
    test_predict = []
    test_sample_label = []
    light_encoder = []
    rgb_encoder = []

    def __init__(self,
                 growth_rate=12,
                 block_config=[(6, 12, 4), (6, 12, 4)],
                 bn_size=4,
                 theta=0.5,
                 num_classes=2):
        super(DensNet, self).__init__()
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.bn_size = bn_size
        self.theta = theta
        self.num_classes = num_classes
        self.myloss = nn.CrossEntropyLoss()
        self.densent = Double_Branch_Dense_Net(num_classes=self.num_classes)
        self.pl_accuracy = torchmetrics.Accuracy()
        self.pl_recall = torchmetrics.Recall(average='none',num_classes=self.num_classes)
        
    def forward(self, X):
        y_hat = self.densent(X)
        return y_hat

    # 3. 定义优化器
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, eps=1e-07)
        return optimizer

    # 4. 训练loop
    def training_step(self, train_batch, batch_ix):
        X, y_true = train_batch
        y_hat, l, r = self.forward(X)
        acc = self.pl_accuracy(toLabel(y_hat), y_true)
        recall = self.pl_recall(toLabel(y_hat),y_true)[self.num_classes - 1]
        loss = self.myloss(y_hat, y_true)
        mylogdict = {
            'loss': loss,
            'log': {
                'train_loss': loss,
                'train_acc': acc,
                'train_recall': recall
            }
        }
        return mylogdict

    # validataion loop
    def validation_step(self, validation_batch, batch_ix):
        X, y_true = validation_batch
        y_hat, l, r = self.forward(X)
        val_acc = self.pl_accuracy(toLabel(y_hat), y_true)
        recall = self.pl_recall(toLabel(y_hat),y_true)[self.num_classes - 1]

        loss = self.myloss(y_hat, y_true)

        self.log_dict({'val_loss': loss, 'val_acc': val_acc, 'recall': recall})
        mylogdict = {
            'log': {
                'val_loss': loss,
                'val_acc': val_acc,
                'val_recall': recall
            }
        }
        return mylogdict

    def test_step(self, test_batch, batch_ix):
        X, y_true = test_batch
        y_hat, l, r = self.forward(X)
        self.test_predict.append(y_hat.cpu())
        self.test_sample_label.append(y_true.cpu())
        self.light_encoder.append(l)
        self.rgb_encoder.append(r)
        return {'test': 'test epoch finish ....'}

    def training_epoch_end(self, output):
        train_loss = sum([out['log']['train_loss'].item()
                          for out in output]) / len(output)
        self.train_epoch_loss.append(train_loss)

        train_acc = sum([out['log']['train_acc'].item()
                         for out in output]) / len(output)
        self.train_epoch_acc.append(train_acc)

        train_recall = sum(
            [out['log']['train_recall'].item()
             for out in output]) / len(output)

    def validation_epoch_end(self, output):
        val_loss = sum([out['log']['val_loss'].item()
                        for out in output]) / len(output)
        self.val_epoch_loss.append(val_loss)

        val_acc = sum([out['log']['val_acc'].item()
                       for out in output]) / len(output)
        val_recall = sum([out['log']['val_recall'].item()
                          for out in output]) / len(output)
        val_loss = sum([out['log']['val_loss'].item()
                        for out in output]) / len(output)

        self.val_epoch_acc.append(val_acc)
        print('val_recall: ', val_recall, '\t', 'mean_val_acc: ', val_acc,
              '\t', 'mean_val_loss: ', val_loss)