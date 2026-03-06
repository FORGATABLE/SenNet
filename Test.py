import SimpleITK as sitk
import os
import numpy as np
import tqdm
train_label_path = r"/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/Dataset103_LiverSeg/labelsTr"
test_label_path = r"/mnt/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/Dataset107_Local_Skin/labelsTr"

# files = os.listdir(train_label_path)
#
# for file in tqdm.tqdm(files):
#     label = sitk.ReadImage(os.path.join(train_label_path, file))
#     label_np = sitk.GetArrayFromImage(label)
#     label_np = np.where(label_np == 5, 1, label_np)
#     label_for_save = sitk.GetImageFromArray(label_np)
#     label_for_save.CopyInformation(label)
#     sitk.WriteImage(label_for_save, os.path.join(train_label_path, file))

files = os.listdir(test_label_path)
for file in tqdm.tqdm(files):
    label = sitk.ReadImage(os.path.join(test_label_path, file))
    label_np = sitk.GetArrayFromImage(label)
    label_np = np.where(label_np ==2, 1, label_np)
    label_for_save = sitk.GetImageFromArray(label_np)
    label_for_save.CopyInformation(label)
    sitk.WriteImage(label_for_save, os.path.join(test_label_path, file))


