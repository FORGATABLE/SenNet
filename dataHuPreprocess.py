import SimpleITK as sitk
import numpy as np
import os
from tqdm import tqdm

train_path = r"/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/Dataset029_LiverAblation2/imagesTr"
test_path = r"/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/Dataset029_LiverAblation2/imagesTs"
output_train = r"/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/Dataset030_LiverAblation2/imagesTr"
output_test = r"/data/zr/Liver_nnUNet_Dataset/nnUNet_raw/Dataset030_LiverAblation2/imagesTs"
if not os.path.exists(output_train):
    os.makedirs(output_train)
if not os.path.exists(output_test):
    os.makedirs(output_test)
train_files = os.listdir(train_path)
test_files = os.listdir(test_path)
train_files = sorted(train_files)
test_files = sorted(test_files)
for file in tqdm(train_files):
    img_name = os.path.join(train_path, file)
    img = sitk.ReadImage(img_name)
    img_data = sitk.GetArrayFromImage(img)
    img_data = np.where(img_data>200, 255, img_data)
    img_data = np.where(img_data<-60, 0, img_data)
    img_data = np.where((img_data>=-60) & (img_data<=200), (img_data+60)*255/260, img_data)
    img_preprocessed = sitk.GetImageFromArray(img_data)
    img_preprocessed.SetOrigin(img.GetOrigin())
    img_preprocessed.SetSpacing(img.GetSpacing())
    img_preprocessed.SetDirection(img.GetDirection())
    sitk.WriteImage(img_preprocessed, os.path.join(output_train, file))
for file in tqdm(test_files):
    img_name = os.path.join(test_path, file)
    img = sitk.ReadImage(img_name)
    img_data = sitk.GetArrayFromImage(img)
    img_data = np.where((img_data>=-60) & (img_data<=200), (img_data+60)*255/260, img_data)
    img_preprocessed = sitk.GetImageFromArray(img_data)
    img_preprocessed.SetOrigin(img.GetOrigin())
    img_preprocessed.SetSpacing(img.GetSpacing())
    img_preprocessed.SetDirection(img.GetDirection())
    sitk.WriteImage(img_preprocessed, os.path.join(output_test, file))
print("finish!!")

