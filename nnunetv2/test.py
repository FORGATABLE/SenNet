import os
import SimpleITK as sitk
import numpy as np
import tqdm
path = r"/mnt/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/Dataset110_FullAndLocalSkin/labelsTr"

files = os.listdir(path)
for file in tqdm.tqdm(files):
    img_path = os.path.join(path, file)
    image = sitk.ReadImage(img_path)
    data = sitk.GetArrayFromImage(image)
    data = np.where(data>0, 1, data)
    new_image = sitk.GetImageFromArray(data)
    new_image.CopyInformation(image)
    sitk.WriteImage(new_image, img_path)

path = r"/mnt/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/Dataset110_FullAndLocalSkin/labelsTs"
files = os.listdir(path)
for file in tqdm.tqdm(files):
    img_path = os.path.join(path, file)
    image = sitk.ReadImage(img_path)
    data = sitk.GetArrayFromImage(image)
    data = np.where(data > 0, 1, data)
    new_image = sitk.GetImageFromArray(data)
    new_image.CopyInformation(image)
    sitk.WriteImage(new_image, img_path)
