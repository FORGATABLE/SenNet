
import os
import time

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from pathlib import Path

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape

##########################################
# ========= 配置部分 ===================== #
##########################################

# coarse / fine 训练输出目录
# 例如: $nnUNet_results/DatasetXYZ/nnUNetTrainer__nnUNetPlans__3d_lowres
COARSE_MODEL_DIR = "/mnt/data/zr/Liver_nnUNet_Dataset/nnUNet_results/Dataset110_FullAndLocalSkin/nnUNetTrainer__nnUNetPlans__3d_lowres"
FINE_MODEL_DIR   = "/mnt/data/zr/Liver_nnUNet_Dataset/nnUNet_results/Dataset112_MaskedFullAndLocalBLL_171/nnUNetTrainer__nnUNetPlans__3d_fullres"

MARGIN = 10
DEVICE = torch.device('cuda',1)

def build_predictor1(model_dir):
    # predictor = nnUNetPredictor(
    #     tile_step_size=0.5,
    #     use_gaussian=True,
    #     use_mirroring=True,
    #     perform_everything_on_device=True,
    #     device=torch.device('cuda', 0),
    #     verbose=False,
    #     verbose_preprocessing=False,
    #     allow_tqdm=True
    # )
    # predictor.initialize_from_trained_model_folder(
    #     join(nnUNet_results, 'Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_lowres'),
    #     use_folds=(0,),
    #     checkpoint_name='checkpoint_final.pth',
    # )
    predictor = nnUNetPredictor(
        tile_step_size = 0.5,
        use_gaussian    = True,
        use_mirroring   = False,
        perform_everything_on_device = True,
        device = DEVICE,
        verbose= False,
        verbose_preprocessing=
        False,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=(0,),                 # single fold
        checkpoint_name="checkpoint_final.pth"
    )
    return predictor
def build_predictor2(model_dir):
    # predictor = nnUNetPredictor(
    #     tile_step_size=0.5,
    #     use_gaussian=True,
    #     use_mirroring=True,
    #     perform_everything_on_device=True,
    #     device=torch.device('cuda', 0),
    #     verbose=False,
    #     verbose_preprocessing=False,
    #     allow_tqdm=True
    # )
    # predictor.initialize_from_trained_model_folder(
    #     join(nnUNet_results, 'Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_lowres'),
    #     use_folds=(0,),
    #     checkpoint_name='checkpoint_final.pth',
    # )
    predictor = nnUNetPredictor(
        tile_step_size = 0.7,
        use_gaussian    = True,
        use_mirroring   = False,
        perform_everything_on_device = True,
        device = DEVICE,
        verbose= False,
        verbose_preprocessing=
        False,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=(0,),                 # single fold
        checkpoint_name="checkpoint_final.pth"
    )
    return predictor


def compute_bbox(mask, margin):
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        raise RuntimeError("Empty ROI from coarse segmentation.")
    zmin,zmax = coords[0].min(), coords[0].max()
    ymin,ymax = coords[1].min(), coords[1].max()
    xmin,xmax = coords[2].min(), coords[2].max()
    return (
        max(0, zmin - margin), min(mask.shape[0], zmax + margin),
        max(0, ymin - margin), min(mask.shape[1], ymax + margin),
        max(0, xmin - margin), min(mask.shape[2], xmax + margin),
    )



def run_inference(in_path, out_path, case_file, coarse_predictor, fine_predictor):
    idx = case_file[:-12]
    case_file = os.path.join(in_path, case_file)
    print("Predict image:" + str(case_file))
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    img, props = SimpleITKIO().read_images([case_file])
    # ret = predictor.predict_single_npy_array(img, props, None, None, False)

    # img = nib.load(case_file)
    # orig_img = img.get_fdata().astype(np.float32)
    orig_img = img
    orig_shape = orig_img[0].shape

    # nnUNet predictor 输入形状: (C, Z, Y, X)
    raw = orig_img

    # ======= coarse 推理 =======
    # predictor 会在内部进行 resample to low_res spacin
    print(f"粗分割耗时为：")
    coarse_seg_lowres = coarse_predictor.predict_single_npy_array(img, props, None, None, False)
    coarse_seg_lowres = coarse_seg_lowres[None]
    print("Finished coarse_seg for image : "+str(case_file))
    # ====== coarse seg 重采样回原始尺寸 ======
    cur_spacing = coarse_predictor.configuration_manager.spacing
    new_spacing = props['spacing']
    coarse_seg_orig = resample_data_or_seg_to_shape(
        coarse_seg_lowres,
        new_shape=orig_img[0].shape,
        current_spacing=cur_spacing,
        new_spacing=new_spacing,
        is_seg=True,
    )

    # ====== 计算原始空间 bbox ======
    bbox = compute_bbox(coarse_seg_orig[0], MARGIN)
    z0,z1,y0,y1,x0,x1 = bbox

    # ====== 裁剪原图 ======
    raw_roi = raw[:, z0:z1, y0:y1, x0:x1]
    print("原本尺寸：" + str(orig_shape))
    tmp = [(int)(bbox[1] - bbox[0]), (int)(bbox[3] - bbox[2]), (int)(bbox[5] - bbox[4])]
    print("bbox尺寸：" + str(tmp))
    # ====== fine 推理 ======
    roi_props = props.copy()
    roi_props["shape_before_cropping"] = orig_img[0].shape
    roi_props["crop_bbox"] = [z0, z1, y0, y1, x0, x1]
    print(f"精分割耗时为：")
    fine_seg_roi = fine_predictor.predict_single_npy_array(raw_roi, roi_props, None, None, False)
    # fine_seg_roi = np.argmax(fine_logits, axis=0)
    # ====== 回贴回原图 ======
    final_seg = np.zeros(orig_shape, dtype=np.uint8)
    final_seg[z0:z1, y0:y1, x0:x1] = fine_seg_roi

    # 保存结果
    out_name = idx + ".nii.gz"
    out_pathname = os.path.join(out_path, out_name)
    # out_pathname.parent.mkdir(exist_ok=True)
    out_nii = SimpleITKIO().write_seg(final_seg, out_pathname, props)

    print(f"Inference completed: {out_pathname}")



def main():
    coarse = build_predictor1(COARSE_MODEL_DIR)
    fine   = build_predictor2(FINE_MODEL_DIR)
    input_path = (r"/mnt/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/"
                  r"Dataset112_MaskedFullAndLocalBLL_171/imagesTs")
    filenames = os.listdir(input_path)
    sorted(filenames)
    output_path = (r"/mnt/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/"
                   r"Dataset112_MaskedFullAndLocalBLL_171/Predict/BLLROIPreFold0Ts")
    os.makedirs(output_path, exist_ok=True)
    i = 0
    for nifti_file in filenames:
        print(f"预测第{i}个CBCT：")
        i = i + 1
        run_inference(input_path, output_path, nifti_file, coarse, fine)

if __name__ == "__main__":
    main()
