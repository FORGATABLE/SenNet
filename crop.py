
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
import numpy as np
import torch.utils.data
# from config import cur_config as cfg
from tqdm import tqdm
import csv
import nibabel as nibs
import pydicom as dcm
import SimpleITK as sitk
import random
# docker 显示中文
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf8" ,line_buffering=True)


patch_size = [20, 256, 256]
half_patch_size = [int( x /2) for x in patch_size]
# 遍历大文件夹中的所有小文件夹，并且把所有小文件夹的名称按顺序存到一个列表中去,得到case_list
import os

data_dir = '/data/zr/train/CT'
label_dir = '/data/zr/train/label' # 需要使用实例标签

save_datadir = '/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/Dataset024_LiverAblation/imagesTr/'
save_labeldir = '/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/Dataset024_LiverAblation/labelsTr/'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# for idx in tqdm(range(420)): # 训练数据
for case_name in os.listdir(data_dir): # 测试数据

    print("--------------------")
    print(case_name)
    print("--------------------")

    # 读取CT数据
    ct_path = f'{data_dir}/{case_name}'
    init_img = sitk.ReadImage(ct_path)
    cur_img = sitk.GetArrayFromImage(init_img)
    # cur_img = torch.from_numpy(cur_img).float().to(device)

    # 读取mask数据
    case_name =case_name[:-12]
    mask_path = f'{label_dir}/{case_name}.nii.gz'
    init_mask = sitk.ReadImage(mask_path)
    cur_mask = sitk.GetArrayFromImage(init_mask)

    unique_values =np.unique(cur_mask)
    unique_values =np.delete(unique_values ,0)
    print("该样本实例标签" ,unique_values)
    # 以每个样本不同的目标为中心切出不同的patch
    for i in unique_values:
        # 此处加载各个消融区域目标的中心点
        mask =np.copy(cur_mask)
        img =np.copy(cur_img)
        # mask = torch.from_numpy(mask).float().to(device)
        img = torch.from_numpy(img).float().to(device)
        # print(case_name)
        # print(mask.shape)
        # print(img.shape)

        idx_x, idx_y, idx_z = np.where(mask == i) # 找到不同的实例区域
        idx_x = torch.from_numpy(idx_x) # 转为tensor类型
        idx_y = torch.from_numpy(idx_y)
        idx_z = torch.from_numpy(idx_z)

        # 最理想的中心分布情况
        # print("中心位置：")
        # print(torch.mean(idx_x.float()),torch.mean(idx_y.float()),torch.mean(idx_z.float()))
        # print(M)
        # x轴
        flag = False
        centroid_x = int(torch.mean(idx_x.float())) # + random.randint(5, 10)
        centroid_y = int(torch.mean(idx_y.float())) # + random.randint(15, 26)
        centroid_z = int(torch.mean(idx_z.float())) # + random.randint(18, 30)
        if (centroid_x <= mask.shape[0] - half_patch_size[0] and
                centroid_x >= half_patch_size[0]):
            print("x中心分布位置理想")
            mask = mask[centroid_x - half_patch_size[0]:centroid_x + half_patch_size[0] ,: ,:]
            img = img[centroid_x - half_patch_size[0]:centroid_x + half_patch_size[0] ,: ,:]
        elif (centroid_x > mask.shape[0] - half_patch_size[0] and
              centroid_x > half_patch_size[0]):
            print("—————————————分布所在区域的x轴深度过高—————————————————") # 中心位置根据实际位置确定
            mask = mask[mask.shape[0] - patch_size[0]:mask.shape[0] ,: ,:]  # 每个轴依次裁剪
            img = img[img.shape[0] - patch_size[0]:img.shape[0] ,: ,:] # 重采样后img的shape与mask同
            # mask = mask[:patch_size,:,:]  # 每个轴依次裁剪
            # img = img[:patch_size,:,:]
        elif (centroid_x < mask.shape[0] - half_patch_size[0] and
              centroid_x < half_patch_size[0]):
            print("—————————————分布所在区域的x轴深度过低—————————————————") # 中心位置根据实际位置确定
            mask = mask[:patch_size[0] ,: ,:]  # 每个轴依次裁剪
            img = img[:patch_size[0] ,: ,:]
            # mask = mask[mask.shape[0] - patch_size:mask.shape[0],:,:]  # 每个轴依次裁剪
            # img = img[mask.shape[0] - patch_size:mask.shape[0],:,:]

        # y轴
        if (centroid_y <= mask.shape[1] - half_patch_size[1] and
                centroid_y >= half_patch_size[1]):
            print("y中心分布位置理想")
            mask = mask[: ,centroid_y - half_patch_size[1]:centroid_y + half_patch_size[1] ,:]
            img = img[: ,centroid_y - half_patch_size[1]:centroid_y + half_patch_size[1] ,:]
        elif (centroid_y > mask.shape[1] - half_patch_size[1] and
              centroid_y > half_patch_size[1]):
            print("—————————————分布所在区域的y轴深度过高—————————————————") # 中心位置根据实际位置确定
            flag = True
            mask = mask[: ,mask.shape[1] - patch_size[1]:mask.shape[1],:]
            img = img[: ,img.shape[1] - patch_size[1]:img.shape[1],:]
            # mask = mask[:,:patch_size,:]
            # img = img[:,:patch_size,:]
        elif (centroid_y < mask.shape[1] - half_patch_size[1] and
              centroid_y < half_patch_size[1]):
            print("—————————————分布所在区域的y轴深度过低—————————————————") # 中心位置根据实际位置确定
            flag = True
            mask = mask[: ,:patch_size[1],:]
            img = img[: ,:patch_size[1],:]
            # mask = mask[:,mask.shape[1] - patch_size:mask.shape[1],:]
            # img = img[:,mask.shape[1] - patch_size:mask.shape[1],:]

        # z轴
        if (centroid_z <= mask.shape[2] - half_patch_size[2] and
                centroid_z >= half_patch_size[2]): # 大于128，小于最大尺度-128
            print("z中心分布位置理想")
            mask = mask[: ,: ,centroid_z - half_patch_size[2]:centroid_z + half_patch_size[2]]
            img = img[: ,: ,centroid_z - half_patch_size[2]:centroid_z + half_patch_size[2]]
        elif (centroid_z > mask.shape[2] - half_patch_size[2] and
              centroid_z > half_patch_size[2]):
            print("—————————————分布所在区域的z轴深度过高—————————————————") # 中心位置根据实际位置确定
            flag = True
            mask = mask[: ,: ,mask.shape[2] - patch_size[2]:mask.shape[2]]
            img = img[: ,: ,img.shape[2] - patch_size[2]:img.shape[2]]
            # mask = mask[:,:,:patch_size]
            # img = img[:,:,:patch_size]
        elif (centroid_z < mask.shape[2] - half_patch_size[2] and
              centroid_z < half_patch_size[2]):
            print("—————————————分布所在区域的z轴深度过低—————————————————") # 中心位置根据实际位置确定
            flag = True
            mask = mask[: ,: ,:patch_size[2]]
            img = img[: ,: ,:patch_size[2]]
            # mask = mask[:,:,mask.shape[2] - patch_size:mask.shape[2]]
            # img = img[:,:,mask.shape[2] - patch_size:mask.shape[2]]

        print("裁剪后的img,mask尺寸" ,img.shape ,mask.shape)

        print("保存语义分割标签")
        mask[mask != 0] = 1  # 有目标的区域标签全都置为1(语义分割标签)
        Mask_path = save_labeldir + 'LiverAblation_' +case_name+'_' + str(round(1)) + '.nii.gz'
        mask = sitk.GetImageFromArray(mask)
        mask.SetSpacing(init_mask.GetSpacing())
        mask.SetOrigin(init_mask.GetOrigin())
        mask.SetDirection(init_mask.GetDirection())
        # if flag:
        #     sitk.WriteImage(mask, Mask_path)
        sitk.WriteImage(mask, Mask_path)

        print("裁剪后的CT尺寸" ,img.shape)
        # 将处理后的img保存
        print("保存CT图像数据")
        img = sitk.GetImageFromArray(img.detach().cpu().numpy())
        img.SetSpacing(init_img.GetSpacing())
        img.SetOrigin(init_img.GetOrigin())
        img.SetDirection(init_img.GetDirection())
        CT_path = save_datadir + 'LiverAblation_' +case_name+'_' + str(round(1)) + '_0000.nii.gz'

        # if flag:
        #     sitk.WriteImage(img, CT_path)
        sitk.WriteImage(img, CT_path)

print("finish")