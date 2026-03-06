"""
   File Name：     EvaluationLiverAblation_voxelnum.py
   Description :   统计预测结果与标签GT区域的体素数
   11.13补充：加入豪斯多夫距离Hausdorff Distance (HD) 指标的计算
"""

# 在得到语义分割结果和边缘检测结果后经后处理得到最终的实例预测结果（针对单个样本的输入，非批量）
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '9' 
import SimpleITK as sitk
import numpy as np
import nibabel as nibs
from tqdm import tqdm
import csv
import torch
import torch.nn.functional as F
import pandas as pd
import surface_distance as surfdist
import GeodisTK
from scipy import ndimage

label_dir = '/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/Dataset024_LiverAblation/labelsTs/' # 语义分割标签，且用于输出所有测试集文件名
output_dir = '/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/Dataset024_LiverAblation/Infer_Foc20Test_1000/' # 只计算其中类别为1的目标分割区域

excel_case_list = []
dice_obj_list = []
dice_edge_list = []
dice_aver_list = []
gt_num_list = []
pred_num_list = []
hd_list = []

file_names = sorted(os.listdir(label_dir), key=lambda x: int(''.join(filter(str.isdigit, x))))
 
for i,filename in enumerate(file_names):
   print(i,filename)
   if filename.endswith(".nii.gz"):
      # 单样本测试
      # filename = 'LymphNode01200520220122.nii.gz'
      fileID = filename.replace(".nii.gz", "") # 将末尾的后缀去掉
      print(fileID,filename)
      fileID = fileID.replace("LiverAblation", "") # 将前缀去掉
      print(fileID,filename)
      excel_case_list.append(fileID)

      pred_path = output_dir + filename
      label_path = label_dir + filename     # 输出和label的文件名相同
      print(pred_path)
      print(label_path)

      # 根据实例分割结果和标签计算预测结果和标签边缘的豪斯多夫距离
      gt_array = nibs.load(label_path).get_fdata().astype(np.float32)
      pred_array = nibs.load(pred_path).get_fdata().astype(np.float32)

      # 统计gt中的非零值作为gt区域体素数
      gt_voxel_num=np.sum(gt_array!=0)
      gt_num_list.append(gt_voxel_num)
      # 统计预测结果中的gt区域体素数
      pred_gtvoxel_num=np.sum(np.bitwise_and(pred_array != 0, gt_array != 0))
      pred_num_list.append(pred_gtvoxel_num)
      print(gt_voxel_num,pred_gtvoxel_num)

print("saving to excel file...")
# 将数据写入excel文件
data = {
    '样本编号': excel_case_list,
    '标签gt区域体素数': gt_num_list,
    '预测结果gt区域体素数':pred_num_list,
}

print(gt_num_list)
print(pred_num_list)

# 创建DataFrame对象
df = pd.DataFrame(data)

# 写入Excel文件
df.to_excel('Dataset024_Foc20Test_1000_voxelnum.xlsx', index=False)