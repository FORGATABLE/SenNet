"""
   File Name:  Eval_Liverablation_SpecialDice.py
   Description: 将预测结果与标签进行对比计算特殊Dice值
                首先将原始的语义分割标签和语义分割预测结果都通过寻找连通域算法获得实例分割标签和实例分割结果。
                在统计的时候，设置对于标签和预测结果中的两区域Dice值大于0.5的为识别到的正确区域,保留该区域。
                没有与任何gt区域的Dice值大于等于0.5的判断为误检区域，将其去除。最后将保留区域后的结果与gt计算Dice作为Special Dice。
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

# 手写豪斯多夫距离计算
def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if dim == 2:
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
    ero = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge

def binary_hausdorff95(s, g, spacing=None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert image_dim == len(g.shape)
    if spacing == None:
        spacing = [1.0] * image_dim
    else:
        assert image_dim==len(spacing)
    img = np.zeros_like(s)
    if image_dim == 2:
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif image_dim == 3:
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1) * 0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    dist2 = dist_list2[int(len(dist_list2) * 0.95)]
    return max(dist1, dist2)

# 单独计算目标区域之间的dice值
def eval_mask_3d(target, predictive, ep=1e-8):
    # 先计算Dice
    predictive = predictive.float()
    target = target.float()

    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    obj_Dice_value = intersection / union

    return obj_Dice_value


label_dir = '/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/Dataset016_LiverAblation/labelsTs/'
output_instance_dir = '/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/Dataset016_LiverAblation/Inference/'

excel_case_list = []
dice_obj_list = []
dice_edge_list = []
dice_aver_list = []
gt_num_list = []
pred_num_list = []
hd_list = []

file_names = sorted(os.listdir(label_dir), key=lambda x: int(''.join(filter(str.isdigit, x))))
print(file_names,len(file_names))

for i,filename in enumerate(file_names):
   print(i,filename)
   if filename.endswith(".nii.gz"):
      # 单样本测试
      # filename = 'LymphNode01200520220122.nii.gz'
      fileID = filename.replace(".nii.gz", "")     # 将末尾的后缀去掉
      print(fileID,filename)
      fileID = fileID.replace("LiverAblation", "") # 将前缀去掉
      print(fileID,filename)
      excel_case_list.append(fileID)

      instance_pred_path = output_instance_dir + filename
      instance_label_path = label_dir + filename 
      
      instance_label = sitk.ReadImage(instance_label_path) # 实例分割标签
      instance_pred = sitk.ReadImage(instance_pred_path) 
      
      output_array = sitk.GetArrayFromImage(instance_pred)
      output_unique_values = np.unique(output_array)

      pred_num = len(output_unique_values)-1
      pred_num_list.append(pred_num)
      print("预测淋巴结个数",pred_num)
      label_array = sitk.GetArrayFromImage(instance_label)
      print(output_array.shape,label_array.shape)
      mask_unique_values = np.unique(label_array)
      # print(mask_unique_values)
      # print("----------")
      label_num = len(mask_unique_values)-1
      gt_num_list.append(label_num)
      print("标签淋巴结个数",label_num)

      # 将最终结果与mask对比，计算每个gt与当前预测出的所有淋巴结目标的Dice(取Dice最高的作为其对应的预测目标，并且将Dice值统计得到平均值)

      mask_unique_values = np.delete(mask_unique_values,0)
      output_unique_values = np.delete(output_unique_values,0)
      # print(mask_unique_values,output_unique_values)
      dice_list = []
      for i,value in enumerate(mask_unique_values):
         # print("value",value)
         temp_mask_array = np.array(list(label_array)).astype(np.float64)
         # 取出每个实例，并把其他的实例像素点的值置0
         temp_mask_array[temp_mask_array != value]=0
         temp_mask_array[temp_mask_array == value]=1
         temp_mask = torch.from_numpy(temp_mask_array) # .cuda()
         # 和预测的所有目标计算Dice
         top_dice = 0

         for i,value1 in enumerate(output_unique_values):
            # print("value1",value1)
            temp_output_array = np.array(list(output_array))
            temp_output_array[temp_output_array != value1]=0
            temp_output_array[temp_output_array == value1]=1
            temp_output_array = temp_output_array.astype(np.float64)
            temp_output = torch.from_numpy(temp_output_array) # .cuda()
            dice_value = eval_mask_3d(temp_mask, temp_output)
            print("dice value",dice_value)
            if dice_value > 0.5: # 被认为是有效预测区域
               print('保留',value1)
               output_array[output_array == value1]=1
    
      # 遍历完所有gt后如果有的区域还不是1，说明其不是有效区域，删除
      output_array[output_array != 1]=0  
      output_array = output_array.astype(np.float64)
      output_tensor = torch.from_numpy(output_array)
      label_array[label_array !=0]=1   # 实例标签->语义标签
      label_array = label_array.astype(np.float64)
      label_tensor = torch.from_numpy(label_array)
      special_dice = eval_mask_3d(label_tensor, output_tensor)     
      print("special_dice值：",special_dice)  
      dice_aver_list.append(round(special_dice.item(),4))

print("saving to excel file...")
# 将数据写入excel文件
data = {
    '样本编号': excel_case_list,
    '实际消融区域个数':gt_num_list,
    '预测消融区域个数':pred_num_list,
    '特殊dice': dice_aver_list,
    '均值': np.mean(dice_aver_list)
}

print(dice_aver_list)
# 创建DataFrame对象
df = pd.DataFrame(data)
df.to_excel('Infer_016_SpecialDice.xlsx', index=False)