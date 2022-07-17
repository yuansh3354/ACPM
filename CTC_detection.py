# =============================================================================
# # impotrs 
# =============================================================================
import os
import cv2
import yaml
import torch
import warnings
import numpy as np
import torchvision
from time import *
import pandas as pd
import torch.nn as nn
from PIL import Image
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import Counter
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')

# =============================================================================
# # my imports
# =============================================================================
from utils.functions import *
from utils.models import *

# =============================================================================
# # get configs
# =============================================================================
config_path = 'Eval_config.yaml'
config_dict = yaml.safe_load(open(config_path, 'r'))

print("setting configs ......")
begin_time = time()
# =============================================================================
# # setting configs 
# =============================================================================
ckpt = config_dict['data_arguments']['ckpt']
theta = config_dict['model_arguments']['theta']
sns_dpi = config_dict['sns_arguments']['sns_dpi']
use_gpu = config_dict['data_arguments']['use_gpu']
bn_size = config_dict['model_arguments']['bn_size']
sns_cmap = config_dict['sns_arguments']['sns_cmap']
input_h5 = config_dict['data_arguments']['input_h5']
sns_title = config_dict['sns_arguments']['sns_title']
sns_labels = config_dict['sns_arguments']['sns_labels']
sns_xlabel = config_dict['sns_arguments']['sns_xlabel']
sns_ylabel = config_dict['sns_arguments']['sns_ylabel']
model_test = config_dict['data_arguments']['model_test']
ctc_result = config_dict['data_arguments']['ctc_result']
growth_rate = config_dict['model_arguments']['growth_rate']
num_classes = config_dict['model_arguments']['num_classes']
sns_save_fig = config_dict['sns_arguments']['sns_save_fig']
block_config = config_dict['model_arguments']['block_config']
sns_figsize = tuple(config_dict['sns_arguments']['sns_figsize'])
check_model_test = config_dict['data_arguments']['check_model_test']

pl_accuracy = torchmetrics.Accuracy()
pl_recall = torchmetrics.Recall(average='none',num_classes=num_classes)

print("Configs setting completed !!!")

print("Predict CTCs ......")

if check_model_test:
	if model_test[0].endswith('csv'):
		X_test = pd.read_csv(model_test[0])
	else:	
		X_test = pd.read_hdf(model_test[0], key=model_test[1][2])
    

if input_h5:
	if input_h5.endswith('csv'):
		X_test = pd.read_csv(input_h5)
	else:	
		X_test = pd.read_hdf(input_h5, key='data')
    
test_dataset = MyDataset(X_test, DataTransforms('test'))
test_dataloader = DataLoader(
    test_dataset, batch_size=64, num_workers=6, shuffle=False)
    
model = DensNet(growth_rate=growth_rate,
                block_config=block_config,
                bn_size=bn_size,
                theta=theta,
                num_classes=num_classes)
model = model.load_from_checkpoint(ckpt,
                                   growth_rate=growth_rate,
                                   block_config=block_config,
                                   bn_size=bn_size,
                                   theta=theta,
                                   num_classes=num_classes)
if use_gpu:
    trainer = pl.Trainer(gpus=-1)
else:
    trainer = pl.Trainer()

model = model_clf(model)

# create test_iter
trainer.test(model, test_dataloader)

# get test accuracy
predict = torch.cat(model.test_predict)
y = torch.cat(model.test_sample_label)
acc = pl_accuracy(toLabel(predict), y)
recall = pl_recall(toLabel(predict),
                                      y)[num_classes - 1]
predict = toLabel(predict)

print(acc, '\n', recall)
# Step.9 Model evaluation(train)
# model clf
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, auc, average_precision_score
model = model_clf(model)

# 评估分类效果
sns.set()
f,ax=plt.subplots()
C2= confusion_matrix(y, predict, labels=sns_labels)
print("Confusion Matrix")
print(C2)
plt.figure(figsize=sns_figsize)
g=sns.heatmap(C2,annot=True,ax=ax,cmap=sns_cmap) #画热力图
ax.set_title(sns_title) #标题
ax.set_xlabel(sns_xlabel) #x轴
ax.set_ylabel(sns_ylabel) #y轴
scatter_fig = g.get_figure()
scatter_fig.savefig(sns_save_fig +'.png', dpi = sns_dpi)
plt.clf()

sns.set()
f,ax=plt.subplots()
C2= confusion_matrix(y, predict, labels=sns_labels)
pC2 = C2 / C2.sum(axis=1)
plt.figure(figsize=sns_figsize)
g=sns.heatmap(pC2,annot=True,ax=ax,cmap=sns_cmap) #画热力图
ax.set_title(sns_title) #标题
ax.set_xlabel(sns_xlabel) #x轴
ax.set_ylabel(sns_ylabel) #y轴
scatter_fig = g.get_figure()
scatter_fig.savefig(sns_save_fig+'_Normalize.png', dpi = sns_dpi)
plt.clf()

assert all(y.numpy() == X_test.label.values),"样本与标签对应错误 (Sample and label corresponding error)"
X_test['predict'] = predict.numpy()
X_test.to_csv(ctc_result)

print("CTCs Predict completed !!!")
end_time = time()
run_time = end_time-begin_time
print ('Running Time：',run_time)
