# imports
import os
import yaml
import warnings
import numpy as np
from time import *
import pandas as pd
from os.path import splitext
import pytorch_lightning as pl
from collections import Counter

warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
# get configs
config_path = 'Train_loop_config.yaml'
config_dict = yaml.safe_load(open(config_path, 'r'))

# setting configs 
data_dir = config_dict['data_arguments']['data_dir']
ext = config_dict['data_arguments']['endwith']
images_info_h5 = config_dict['data_arguments']['save_files']['images_info_h5']
rgb_light_h5 = config_dict['data_arguments']['save_files']['rgb_light_h5']
use_image_type = config_dict['data_arguments']['use_image_type']
labels = config_dict['data_arguments']['labels']

print("Read files......")
begin_time = time()

# Step-01 get data info 
# Read all images in target-file
# Note: plz stored each samples in independent flods, and also make sure that each cell-type is stored seprately
# 循环读取目标文件夹下的所有文件
# 注意，每个样本单独存放，每个样本对应的 CK++/CTC（以及其他） 的细胞类型也单独存放
patient_path = [os.path.join(data_dir,i) for i in os.listdir(data_dir)]
my_list = []
for patient in patient_path:
    for j in os.listdir(patient):
        image = os.path.join(patient,j)
        image_path = [os.path.join(image,i) for i in os.listdir(image)]
        my_list = my_list + image_path

print("Data read completed !!!")


# Data Clean
# Get sample_id, cell-type, image_id, image_type
# 数据清洗
# 提取样本id，细胞类型，图片（细胞）id，图片类型
print("Data Cleaning & save images infomation ......")
my_list_cl = [i for i in my_list if i.lower().endswith(tuple(ext))]       
ids = len(data_dir.split('/'))
patient_id = [i.split('/')[ids] for i in my_list_cl]
cell_type = [i.split('/')[ids+1] for i in my_list_cl]
image_id = [i.split('/')[ids+2] for i in my_list_cl]
image_type = [splitext(splitext(i)[0])[1].split('.')[1].upper() for i in my_list_cl]

# To DataFrame
# 构建矩阵
df = pd.DataFrame({'image_id':image_id,
              'patient_id':patient_id,
              'cell_type':cell_type,
              'image_path':my_list_cl,
            'image_type':image_type
             })

# save .h5 file 
# 保存数据并输出
h5 = pd.HDFStore(images_info_h5,'w', complevel=4, complib='blosc')
h5['data'] = df
h5.close()



end_time = time()
run_time = end_time-begin_time
print ('Complete save all images information, Running Time ：',run_time)


print("Reload files to get RBG-Light images ......")
begin_time = time()

# Reload files, and get RGB-light images
# 导入数据，提取RGB和亮场图
df = pd.read_hdf(images_info_h5,key='data')
my_list = df.image_id.values
my_list_cl = [i for i in my_list if splitext(i)[0].lower().endswith(tuple(use_image_type))]
df = df.loc[df.image_id.isin(my_list_cl)]

# Get ng_CTC and p_CTC
# 提取所需样本
pl.utilities.seed.seed_everything(seed=42)
c4 = df.loc[df['cell_type'] == labels[0]]
c6 = df.loc[df['cell_type'] == labels[1]]
df = pd.concat([c4,c6])

# Get labels to numeric
df['label'] = 0
df.loc[df.cell_type == labels[1],'label'] = 1

# Get RGB & light seprately
rgb = df.loc[df['image_type'] == use_image_type[0].upper()]
light = df.loc[df['image_type'] == use_image_type[1].upper()]

print("RBG-Light Data Clean ......")

# Make sure No duplicate
# 剔除所有重复样本
cell_id = [splitext(splitext(i)[0])[0] for i in rgb.image_id.values]
rgb['cell_id'] = cell_id

cell_id = [splitext(splitext(i)[0])[0] for i in light.image_id.values]
light['cell_id'] = cell_id

print('Number of light-sample before clean:{}'.format(light.shape[0]))
ids = light.groupby('cell_id').count()>1
price = ids[ids['label'] == True].index
light = light.loc[~light['cell_id'].isin(price),]
print('Number of light-sample after clean: {}'.format(light.shape[0]))

print('Number of rgb-sample before clean: {}'.format(rgb.shape[0]))
ids = rgb.groupby('cell_id').count()>1
price = ids[ids['label'] == True].index
rgb = rgb.loc[~rgb['cell_id'].isin(price),]
print('Number of rgb-sample after clean: {}'.format(rgb.shape[0]))

rgb.index = rgb.cell_id.values
light.index = light.cell_id.values
ids = rgb.index.intersection(light.index)
print('Number of same sample of Light-RGB: {}'.format(len(ids)))
rgb = rgb.loc[ids]
light = light.loc[ids]

all( rgb.index == light.index)

rgb['ligth_path'] = light.image_path
df = rgb.copy()
df = df[[ 'cell_id','label' ,'image_path','ligth_path','image_id', 'patient_id', 'cell_type',  'image_type']]

# save RGB_light image informations
h5 = pd.HDFStore(rgb_light_h5,'w', complevel=4, complib='blosc')
h5['data'] = df
h5.close()

print('Shape of RBG-Light DataFrame: ',df.shape)
print('Data Distribution:',Counter(df.cell_type.values))
print('Data Summary:')
print(df.head())

end_time = time()
run_time = end_time-begin_time
print ('Complete save RBG-Light information, Running Time ：',run_time)
