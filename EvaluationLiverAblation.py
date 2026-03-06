"""
   File Name：     EvaluationLiverAblation.py
   Description :   将预测结果与标签进行对比计算Dice值
   11.13补充：加入豪斯多夫距离Hausdorff Distance (HD) 指标的计算
"""

# 在得到语义分割结果和边缘检测结果后经后处理得到最终的实例预测结果（针对单个样本的输入，非批量）
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
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
    if len(dist_list2)==0:
        return 0
    else:
        dist2 = dist_list2[int(len(dist_list2) * 0.95)]
        return max(dist1, dist2)

# 单独计算目标区域之间的dice值
def eval_mask_3d(target, predictive, ep=1e-8):
    # 先计算Dice
    # a = torch.tensor(0).cuda()
    # b = torch.tensor(1).cuda()
    # predictive = torch.where(predictive == 0, a, b) # 有目标的部分全部置为1
    predictive = predictive.float()
    target = target.float()

    # print(target.type(), predictive.type())
    intersection = 2 * torch.sum(predictive * target) + ep
    # print(intersection)
    union = torch.sum(predictive) + torch.sum(target) + ep
    # print(union)
    obj_Dice_value = intersection / union

    return obj_Dice_value

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
      pred_img = nibs.load(pred_path)
      pred_array = pred_img.get_fdata().astype(np.float32)
      spacing = pred_img.header.get_zooms()

      # 可以使用该手写函数（已经包含边缘的提取） 也可以使用下方库的函数
      hd_dist_95 = binary_hausdorff95(pred_array, gt_array, spacing=spacing)
      print("边界豪斯多夫距离",hd_dist_95)
      hd_list.append(round(hd_dist_95,4))

      label_tensor = torch.from_numpy(gt_array)
      pred_tensor = torch.from_numpy(pred_array)

      obj_dice_value = eval_mask_3d(label_tensor,pred_tensor)
      
      print("该样本目标分割dice值",str(round(obj_dice_value.item(),4)))
      dice_obj_list.append(round(obj_dice_value.item(),4))

print("saving to excel file...")
# 将数据写入excel文件
data = {
    '样本编号': excel_case_list,
    '目标分割Dice':dice_obj_list,
    '边界豪斯多夫距离':hd_list,
    'Dice平均值':np.mean(dice_obj_list),
    '豪斯多夫平均值':np.mean(hd_list)
}

print(dice_obj_list)
print(hd_list)

# 创建DataFrame对象
df = pd.DataFrame(data)

# 写入Excel文件
df.to_excel('Dataset024_Foc20Test_1000.xlsx', index=False)