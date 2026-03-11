"""
   File Name：     eval.py
   Description :   对比标签和预测结果，计算dice，hausdrone距离, IOU, prediction, recall目标体素数等指标
"""

# 在得到语义分割结果和边缘检测结果后经后处理得到最终的实例预测结果（针对单个样本的输入，非批量）
import os
# 指定使用的 GPU号
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import SimpleITK as sitk
import numpy as np
import nibabel as nibs
from typing import Union, Optional
from tqdm import tqdm
import csv
import torch
import torch.nn.functional as F
import pandas as pd
from typing import List, Union, Tuple
# import GeodisTK
from scipy import ndimage
from scipy.ndimage import binary_erosion, distance_transform_edt


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
    ero = binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge


# def binary_hausdorff95(s, g, spacing=None):
#     """
#     get the hausdorff distance between a binary segmentation and the ground truth
#     inputs:
#         s: a 3D or 2D binary image for segmentation
#         g: a 2D or 2D binary image for ground truth
#         spacing: a list for image spacing, length should be 3 or 2
#     """
#     s_edge = get_edge_points(s)
#     g_edge = get_edge_points(g)
#     image_dim = len(s.shape)
#     assert image_dim == len(g.shape)
#     if spacing == None:
#         spacing = [1.0] * image_dim
#     else:
#         assert image_dim == len(spacing)
#     img = np.zeros_like(s)
#     if image_dim == 2:
#         s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
#         g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
#     elif image_dim == 3:
#         s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
#         g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)
#
#     dist_list1 = s_dis[g_edge > 0]
#     dist_list1 = sorted(dist_list1)
#     if len(dist_list1) == 0:
#         dist1 = 0
#         dist3 = 0
#     else:
#         dist1 = dist_list1[int(len(dist_list1) * 0.95)]
#         dist3 = dist_list1[int(len(dist_list1))-1]
#     dist_list2 = g_dis[s_edge > 0]
#     dist_list2 = sorted(dist_list2)
#     if len(dist_list2) == 0:
#         return 0, 0
#     else:
#         dist2 = dist_list2[int(len(dist_list2) * 0.95)]
#         dist4 = dist_list2[int(len(dist_list2))-1]
#         return max(dist1, dist2), max(dist3, dist4)
def binary_hausdorff95(s, g, spacing=None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 3D or 2D binary image for ground truth
        spacing: a list/tuple for image spacing, length should be 3 or 2
    returns:
        hd95, hd100
    """
    s = np.asarray(s).astype(bool)
    g = np.asarray(g).astype(bool)

    image_dim = len(s.shape)
    assert image_dim == len(g.shape)

    if spacing is None:
        spacing = [1.0] * image_dim
    else:
        assert image_dim == len(spacing)

    s_edge = get_edge_points(s).astype(bool)
    g_edge = get_edge_points(g).astype(bool)

    if not np.any(s_edge) and not np.any(g_edge):
        return 0.0, 0.0
    if not np.any(s_edge) or not np.any(g_edge):
        return float("inf"), float("inf")

    s_dist_map = distance_transform_edt(~s_edge, sampling=spacing)
    g_dist_map = distance_transform_edt(~g_edge, sampling=spacing)

    dist_list1 = s_dist_map[g_edge]
    dist_list2 = g_dist_map[s_edge]

    if len(dist_list1) == 0 and len(dist_list2) == 0:
        return 0.0, 0.0
    if len(dist_list1) == 0 or len(dist_list2) == 0:
        return float("inf"), float("inf")

    hd95_1 = np.percentile(dist_list1, 95)
    hd95_2 = np.percentile(dist_list2, 95)
    hd_1 = np.max(dist_list1)
    hd_2 = np.max(dist_list2)

    return float(max(hd95_1, hd95_2)), float(max(hd_1, hd_2))
def getBestAndWorst(metrics_list):
    """
    :param metrics_list: list of metrics
    :return: average, best, worst, 25, 75
    """
    average = round(sum(metrics_list) / len(metrics_list), 4)
    best = max(metrics_list)
    worst = min(metrics_list)
    metrics_list.sort()
    n = len(metrics_list)
    q25 = metrics_list[int(n*0.25)]
    q75 = metrics_list[int(n*0.75)]
    return average, best, worst, q25, q75



# 单独计算目标区域之间的dice值
# def eval_mask_3d(target, predictive, ep=1e-8):
#     # 先计算Dice
#     # a = torch.tensor(0).cuda()
#     # b = torch.tensor(1).cuda()
#     # predictive = torch.where(predictive == 0, a, b) # 有目标的部分全部置为1
#     predictive = predictive.float()
#     target = target.float()
#
#     # print(target.type(), predictive.type())
#     intersection = 2 * torch.sum(predictive * target) + ep
#     # print(intersection)
#     union = torch.sum(predictive) + torch.sum(target) + ep
#     # print(union)
#     obj_Dice_value = intersection / union
#
#     return obj_Dice_value
def eval_metrics_3d(target, predictive, epsilon=1e-8):
    """
    计算3D分割任务的Dice、IOU、Recall、Precision
    输入:
        target (torch.Tensor): 真实标签（需为0或1，或可阈值化的概率值）
        predictive (torch.Tensor): 预测结果（需为概率值或已二值化的标签）
        epsilon (float): 平滑项，防止除零错误
    输出:
        dice, iou, recall, precision (tuple)
    """
    # 二值化预测结果
    pred_binary = predictive.float()
    target_binary = target.float()

    # 计算交集(TP)、GT和Pred的总像素数
    intersection = torch.sum(predictive * target)
    sum_gt = torch.sum(target)
    sum_pred = torch.sum(predictive)

    # 计算并集、FP、FN
    union = sum_gt + sum_pred - intersection
    fp = sum_pred - intersection  # 预测为1但GT为0的像素数
    fn = sum_gt - intersection  # GT为1但预测为0的像素数

    # 计算指标（添加平滑项）
    dice = (2. * intersection + epsilon) / (sum_gt + sum_pred + epsilon)
    iou = (intersection + epsilon) / (union + epsilon)
    recall = (intersection + epsilon) / (sum_gt + epsilon)  # TP / (TP + FN)
    precision = (intersection + epsilon) / (sum_pred + epsilon)  # TP / (TP + FP)

    return dice, iou, recall, precision

def get_mask(segmentation:np.ndarray, region_or_label:int) ->np.ndarray:
    """
    得到不同标签的Mask
    """
    mask = np.zeros_like(segmentation)
    mask[segmentation == region_or_label] = True
    return mask

def get_single_dice(reference_file:str, prediction_file:str, labels_or_regions:List[int]) ->dict:

    seg_ref_img = nibs.load(reference_file)
    seg_ref = seg_ref_img.get_fdata().astype(np.float32)
    seg_pred = nibs.load(prediction_file).get_fdata().astype(np.float32)
    spacing = seg_ref_img.header.get_zooms()

    results = {}
    for r in labels_or_regions:
        mask_ref = get_mask(seg_ref, r)
        mask_pred = get_mask(seg_pred, r)
        hd_95, hd = binary_hausdorff95(mask_pred, mask_ref, spacing=spacing)


        label_tensor = torch.from_numpy(mask_ref)
        pred_tensor = torch.from_numpy(mask_pred)
        dice, iou, recall, precision = eval_metrics_3d(label_tensor, pred_tensor)
        print(f"label{r}的dice值为{round(dice.item(), 4)}")
        results[f'label{r}_dice'] = str(round(dice.item(), 4))
        results[f'label{r}_iou'] = str(round(iou.item(), 4))
        results[f'label{r}_recall'] = str(round(recall.item(), 4))
        results[f'label{r}_precision'] = str(round(precision.item(), 4))
        results[f'label{r}_hd'] = str(round(hd, 4))
        results[f'label{r}_hd_95'] = str(round(hd_95, 4))

    # 统计GT中的非零值作为GT区域体素数
    gt_voxel_num = np.sum(seg_ref != 0)
    results['gt_num_list'] = gt_voxel_num
    if seg_pred.shape == seg_ref.shape:
        pred_gtvoxel_num = np.sum(np.bitwise_and(seg_ref != 0, seg_pred != 0))
        results['pred_num_list'] = pred_gtvoxel_num
        print(gt_voxel_num, pred_gtvoxel_num)
    else:
        results['pred_num_list'] = '/'

    return results




class My_eval():
    def __init__(self, label_dir, pred_dir, metrics1, metrics2, labels_or_regions:List[int]):
        self.label_dir = label_dir
        self.pred_dir = pred_dir
        self.metrics1 = metrics1
        self.metrics2 = metrics2
        self.labels_or_regions = labels_or_regions


    def compute_multi_dice(self):
        data = {}
        data['cases'] = []
        for i in self.labels_or_regions:
            data[f'label{i}_dice'] = []
            data[f'label{i}_iou'] = []
            data[f'label{i}_recall'] = []
            data[f'label{i}_precision'] = []
            data[f'label{i}_hd'] = []
            data[f'label{i}_hd_95'] = []
        data['gt_num_list'] = []
        data['pred_num_list'] = []
        file_names = sorted(os.listdir(self.label_dir), key=lambda x: int(''.join(filter(str.isdigit, x))))
        # file_names2 = file_names[:10]
        for i, file in enumerate(file_names):
            print(i, file)
            if file.endswith(".nii.gz"):
                # 单样本测试
                # filename = 'LymphNode01200520220122.nii.gz'
                fileID = file.replace(".nii.gz", "")
                fileID = fileID.replace("cbct_", "")
                print(fileID, file)
                data['cases'].append(fileID)

                label_path = self.label_dir + file
                pred_path = self.pred_dir + file
                print(label_path)
                print(pred_path)

                result = get_single_dice(label_path, pred_path, self.labels_or_regions)
                for k, v in result.items():
                    data[k].append(v)

        print("saving to excel file...")
        # 创建DataFrame对象
        df = pd.DataFrame(data)
        # 写入Excel文件
        df.to_excel(self.metrics1, index=False)

    def Compute_dice(self):
        excel_case_list = []
        dice_obj_list = []
        gt_num_list = []
        pred_num_list = []
        hd_list95 = []
        hd_list = []
        IOU_list = []
        recall_list = []
        precise_list = []

        file_names = sorted(os.listdir(self.label_dir), key=lambda x: int(''.join(filter(str.isdigit, x))))

        for i, filename in enumerate(file_names):
            print(i, filename)
            if filename.endswith(".nii.gz"):
                # 单样本测试
                # filename = 'LymphNode01200520220122.nii.gz'
                fileID = filename.replace(".nii.gz", "")  # 将末尾的后缀去掉
                fileID = fileID.replace("LiverAblation", "")  # 将前缀去掉
                print(fileID, filename)
                excel_case_list.append(fileID)
                pred_path = self.pred_dir + filename
                label_path = self.label_dir + filename  # 输出和label的文件名相同
                print(pred_path)
                print(label_path)
                gt_array = nibs.load(label_path).get_fdata().astype(np.float32)
                pred_img = nibs.load(pred_path)
                pred_array = pred_img.get_fdata().astype(np.float32)
                spacing = pred_img.header.get_zooms()

                # 根据实例分割结果和标签计算预测结果和标签边缘的豪斯多夫距离
                hd_dist_95, hd_dist = binary_hausdorff95(pred_array, gt_array, spacing=spacing)
                print("边界豪斯多夫距离95:", hd_dist_95)
                print("边界豪斯多夫距离:", hd_dist)
                hd_list95.append(round(hd_dist_95, 4))
                hd_list.append(round(hd_dist, 4))

                label_tensor = torch.from_numpy(gt_array)
                pred_tensor = torch.from_numpy(pred_array)

                # 可以使用该手写函数（已经包含边缘的提取） 也可以使用下方库的函数
                # obj_dice_value = eval_mask_3d(label_tensor, pred_tensor)
                obj_dice_value, Iou, recall, precise = eval_metrics_3d(label_tensor, pred_tensor)
                print("该样本目标分割dice值", str(round(obj_dice_value.item(), 4)))
                print("该样本目标分割IOU值", str(round(Iou.item(), 4)))
                print("该样本目标分割Recall值", str(round(recall.item(), 4)))
                print("该样本目标分割Precise值", str(round(precise.item(), 4)))

                dice_obj_list.append(round(obj_dice_value.item(), 4))
                IOU_list.append(round(Iou.item(), 4))
                recall_list.append(round(recall.item(), 4))
                precise_list.append(round(precise.item(), 4))

                # 计算消融区域的体素点
                # 统计gt中的非零值作为gt区域体素数
                gt_voxel_num = np.sum(gt_array != 0)
                gt_num_list.append(gt_voxel_num)
                # 统计预测结果中的gt区域体素数
                if pred_array.shape == gt_array.shape:
                    pred_gtvoxel_num = np.sum(np.bitwise_and(pred_array != 0, gt_array != 0))
                    pred_num_list.append(pred_gtvoxel_num)
                    print(gt_voxel_num, pred_gtvoxel_num)
                else:
                    pred_num_list.append('/')



        print("saving to excel file...")
        # 将数据写入excel文件
        data = {
            '样本编号': excel_case_list,
            '目标分割Dice': dice_obj_list,
            '目标分割IOU':IOU_list,
            '边界豪斯多夫距离':hd_list,
            '边界豪斯多夫距离95': hd_list95,
            'recall':recall_list,
            'precise':precise_list,
            'GT体素数': gt_num_list,
            '预测体素数': pred_num_list
        }

        # print(dice_obj_list)
        # print(hd_list)

        # 创建DataFrame对象
        df = pd.DataFrame(data)

        # 写入Excel文件
        df.to_excel(self.metrics1, index=False)


    def get_multi_metrics(self):
        results = {'分割': ['平均', '最佳', '最差', '75分位', '25分位']}
        skip = ['cases', 'gt_num_list', 'pred_num_list']
        data = pd.read_excel(self.metrics1)
        columns = data.columns.tolist()
        for col in columns:
            if col not in skip:
                print(col)
                eval_list = data[col].tolist()
                # 计算每个指标的平均值、最佳值、最差值、75分位、25分位
                sorted(eval_list)
                if 'hd' in col:
                    temp_average, temp_worst, temp_best, temp_75, temp_25 = getBestAndWorst(eval_list)
                else:
                    temp_average, temp_best, temp_worst, temp_25, temp_75 = getBestAndWorst(eval_list)
                results[col] = [temp_average, temp_best, temp_worst, temp_75, temp_25]
        # 将结果转换为DataFrame
        df = pd.DataFrame(results)
        # 将结果保存到Excel文件
        df.to_excel(self.metrics2, index=False)



    def get_metrics(self):
        data = pd.read_excel(self.metrics1)
        data = data.iloc[:, 1:7]
        dice_list = data.iloc[:, 0].tolist()
        IOU_list = data.iloc[:, 1].tolist()
        hd_list = data.iloc[:, 2].tolist()
        hd_list95 = data.iloc[:, 3].tolist()
        recall_list = data.iloc[:, 4].tolist()
        precision_list = data.iloc[:, 5].tolist()

        sorted(dice_list)
        sorted(IOU_list)
        sorted(hd_list, reverse=True)
        sorted(hd_list95, reverse=True)
        sorted(recall_list)
        sorted(precision_list)

        dice_average, dice_best, dice_worst, dice_25, dice_75 = getBestAndWorst(dice_list)
        IOU_average, IOU_best, IOU_worst, IOU_25, IOU_75 = getBestAndWorst(IOU_list)
        hd_average, hd_worst, hd_best, hd_25, hd_75 = getBestAndWorst(hd_list)
        hd95_average, hd95_worst, hd95_best, hd95_25, hd95_75 = getBestAndWorst(hd_list95)
        recall_average, recall_best, recall_worst, recall_25, recall_75 = getBestAndWorst(recall_list)
        precision_average, precision_best, precision_worst, precision_25, precision_75 = getBestAndWorst(precision_list)

        dice = [dice_average, dice_best, dice_worst, dice_25, dice_75]
        IOU = [IOU_average, IOU_best, IOU_worst, IOU_25, IOU_75]
        hd = [hd_average, hd_best, hd_worst, hd_25, hd_75]
        hd95 = [hd95_average, hd95_best, hd95_worst, hd95_25, hd95_75]
        recall = [recall_average, recall_best, recall_worst, recall_25, recall_75]
        precision = [precision_average, precision_best, precision_worst, precision_25, precision_75]
        metrics = ['平均', '最佳', '最差', '75分位', '25分位']

        results = {
            '消融区域': metrics,
            'Dice': dice,
            'IOU': IOU,
            'HD': hd,
            'HD95': hd95,
            'Precision': precision,
            'Recall': recall
        }
        df = pd.DataFrame(results)
        df.to_excel(self.metrics2, index=False)


# labe l_dir = '/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/Dataset024_LiverAblation/labelsTs/'  # 语义分割标签，且用于输出所有测试集文件名
# output_dir ='/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/Dataset024_LiverAblation/Infer_Foc20Test_1000/'  # 只计算其中类别为1的目标分割区域
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算目标分割的dice值")

    # 定义命令行参数
    parser.add_argument('--gt', type=str, required=True, help='真实标签（Ground Truth）目录路径')
    parser.add_argument('--pre', type=str, required=True, help='预测结果目录路径')
    parser.add_argument('--metrics1', type=str, required=True, help='每样本指标保存路径（xlsx格式）')
    parser.add_argument('--metrics2', type=str, required=True, help='总体指标保存路径, '
                                                '其中包括总体指标的平均值、最佳值、最差值、75分位、25分位（xlsx格式）')
    args = parser.parse_args()
    labels_or_regions = [1, 2, 3, 4]
    # labels_or_regions = [1]
    evaluator = My_eval(args.gt, args.pre, args.metrics1,
                        args.metrics2, labels_or_regions)
    evaluator.compute_multi_dice()
    evaluator.get_multi_metrics()
    # evaluator.get_metrics()











