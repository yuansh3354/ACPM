# impotrs 
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

# my imports
from utils.functions import *
from utils.models import *

# get configs
config_path = 'configs/Train_loop_config.yaml'
config_dict = yaml.safe_load(open(config_path, 'r'))

print("setting configs ......")

# setting configs 
theta = config_dict['model_arguments']['theta']
labels = config_dict['data_arguments']['labels']
bn_size = config_dict['model_arguments']['bn_size']
ckpt_dir = config_dict['model_arguments']['ckpt_dir']
test_size = config_dict['data_arguments']['test_size']
ckpt_name = config_dict['model_arguments']['ckpt_name']
output_dir = config_dict['model_arguments']['output_dir']
save_top_k = config_dict['model_arguments']['save_top_k']
pl_monitor = config_dict['model_arguments']['pl_monitor']
random_seed = config_dict['golbal_setting']['random_seed']
growth_rate = config_dict['model_arguments']['growth_rate']
num_classes = config_dict['model_arguments']['num_classes']
downsampling = config_dict['data_arguments']['downsampling']
max_epochs = config_dict['training_arguments']['max_epochs']
block_config = config_dict['model_arguments']['block_config']
num_workers = config_dict['training_arguments']['num_workers']
downsample_rate = config_dict['data_arguments']['downsample_rate']
validation_size = config_dict['data_arguments']['validation_size']
pl_monitor_mode = config_dict['model_arguments']['pl_monitor_mode']
test_batch_size = config_dict['training_arguments']['test_batch_size']
train_batch_size = config_dict['training_arguments']['train_batch_size']
rgb_light_clean = config_dict['data_arguments']['save_files']['rgb_light_h5']
independent_test = config_dict['data_arguments']['save_files']['independent_test']
trainig_loop_data_set = config_dict['data_arguments']['save_files']['trainig_loop_data_set']

print("Configs setting completed !!!")

print("Dataloading ......")
begin_time = time()
df = pd.read_hdf(rgb_light_clean, key='data')
print('Data Distribution before downsampling {}'.format(
    Counter(df.label.values)))

pl.utilities.seed.seed_everything(seed=random_seed)
num_neg_CTCs = Counter(df['cell_type'].values)[labels[0]]
if downsampling:
    idx = int(num_neg_CTCs * downsample_rate)
    c4 = df.loc[df['cell_type'] == labels[0]].sample(idx)
else:
    c4 = df.loc[df['cell_type'] == labels[0]]
c6 = df.loc[df['cell_type'] == labels[1]]

df = pd.concat([c4, c6])
print('Data Distribution after downsampling {}'.format(Counter(
    df.label.values)))

if downsampling:
    in_df = pd.read_hdf(rgb_light_clean, key='data')
    in_df = in_df.loc[~in_df.image_path.isin(df.image_path.values), ]
    h5 = pd.HDFStore(independent_test,'w', complevel=4, complib='blosc')
    h5['data'] = in_df
    h5.close()
    print('独立样本分布 {}'.format(Counter(in_df.label.values)))

print("Data split ......")
print('Test data ratio: ',test_size )
print('validation data ratio: ',validation_size )
X_train, X_test= train_test_split(
    df,
    test_size=test_size,  # 验证集比例
    random_state=random_seed,  # 随机种子
    stratify=df.label  #分层抽样
)

X_train, X_val= train_test_split(
    X_train,
    test_size=validation_size,  # 验证集比例
    random_state=random_seed,  # 随机种子
    stratify=X_train.label  #分层抽样
)
print("Save training data ......")
h5 = pd.HDFStore(trainig_loop_data_set[0],'w', complevel=4, complib='blosc')
h5[trainig_loop_data_set[1][0]] = X_train
h5[trainig_loop_data_set[1][1]] = X_val
h5[trainig_loop_data_set[1][2]] = X_test
h5.close()
print("training data save completed !!!")

# dataloader
pl.utilities.seed.seed_everything(seed=random_seed)
train_dataset = MyDataset(X_train, DataTransforms('train'))
train_dataloader = DataLoader(
    train_dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)

pl.utilities.seed.seed_everything(seed=random_seed)
val_dataset = MyDataset(X_val, DataTransforms('test'))
val_dataloader = DataLoader(
    val_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=True)

pl.utilities.seed.seed_everything(seed=random_seed)
test_dataset = MyDataset(X_test, DataTransforms('test'))
test_dataloader = DataLoader(
    test_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False)

print("Model Creating ......")
model = DensNet(growth_rate=growth_rate,
                block_config=block_config,
                bn_size=bn_size,
                theta=theta,
                num_classes=num_classes)


OUTPUT_DIR = output_dir
tb_logger = pl.loggers.TensorBoardLogger(save_dir='./', name=ckpt_dir)
# set check point to choose best model
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=tb_logger.log_dir,
    filename=ckpt_name,
    save_top_k=save_top_k,  # 保留最优的15
    monitor=pl_monitor,  # check acc
    mode=pl_monitor_mode)  # 参数保存位置

trainer = pl.Trainer(gpus=-1,
                     callbacks=[checkpoint_callback],
                     max_epochs=max_epochs)  # 开始训练
trainer.fit(model, train_dataloader, val_dataloader)  # 训练模型

end_time = time()
run_time = end_time-begin_time
print ('Complete save RBG-Light information, Running Time ：',run_time)




























































