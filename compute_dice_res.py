# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

path = r"/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/Dataset028_LiverAblation1/resultsPre2Fold01.xlsx"
data = pd.read_excel(path)
print(data.shape)
# 取第二列到第四列
data = data.iloc[0:20, [1,2,3]]
# 将三列转化为分别三列列表
dice_list = data.iloc[:, 0].tolist()
# print(dice_list)
Hausdorff_list = data.iloc[:, 1].tolist()

pixel_list = data.iloc[:, 2].tolist()
# print(pixel_list)

dice_res = 0
hausedroff_res = 0
for i in range(len(pixel_list)):
    weight = pixel_list[i] / sum(pixel_list)
    hausedroff_res += Hausdorff_list[i] * weight
    dice_res += dice_list[i] * weight

# 四位小数输出
print(f'wighted dice:{dice_res:.4f}')
print(f'wighted Hausdorff:{hausedroff_res:.4f}')
print(f'mean dice :{np.mean(dice_list):.4f}')
print(f'mean Hausdorff:{np.mean(Hausdorff_list):.4f}')