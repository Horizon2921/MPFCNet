from monai.transforms import (
    AddChanneld,
    LoadImage,
    LoadImaged,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
)
from monai.config import print_config
from monai.apps import download_and_extract
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob

# 此为一个单独的.py文件，存储各种路径信息和参数
from init import Options
opt = Options().parse()

# 读取路径中的文件，生成字典列表
train_images = sorted(
    glob.glob(os.path.join(opt.images_folder, "*.nii.gz")))
train_labels = sorted(
    glob.glob(os.path.join(opt.labels_folder, "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]
# 分割字典列表中的图像数据，倒数10个作为val数据，剩下的都是训练数据
train_dicts, val_dicts = data_dicts[:-10], data_dicts[-10:]

print(train_dicts[0])   # 输出数据的路径和名称

# 读取Nifti 文件，返回图像数据数组 arrays 以及元数据metadata(包括放射信息和体素大小）
loader = LoadImage(dtype=np.float32, image_only=True)
image = loader(train_dicts[0]["image"])
# print(f"input: {train_data_dicts[0]['image']}")
print(f"image shape: {image.shape}")
print(f"image affine:\n{image.meta['affine']}")
print(f"image pixdim:\n{image.pixdim}")

# LoadImaged是相应的基于dict的LoadImage版本
loader2 = LoadImaged(keys=("image", "label"), image_only=False)
data_dict = loader2(train_dicts[0])
# print(f"input:, {train_dicts[0]}")
print(f"image shape: {data_dict['image'].shape}")
print(f"label shape: {data_dict['label'].shape}")
print(f"image pixdim:\n{data_dict['image'].pixdim}")

# image, label = data_dict["image"], data_dict["label"]
# plt.figure("visualize", (8, 4))
# plt.subplot(1, 2, 1)
# plt.title("image")
# plt.imshow(image[:, :, 60], cmap="gray")
# plt.subplot(1, 2, 2)
# plt.title("label")
# plt.imshow(label[:, :, 60])
# plt.show()

orientation = Orientationd(keys=["image", "label"], axcodes="PLI")
data_dict = orientation(data_dict)
print(f"image shape: {data_dict['image'].shape}")
print(f"label shape: {data_dict['label'].shape}")
print(f"image affine after Spacing:\n{data_dict['image'].meta['affine']}")
print(f"label affine after Spacing:\n{data_dict['label'].meta['affine']}")

image, label = data_dict["image"], data_dict["label"]
plt.figure("visualise", (8, 4))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[ :, :, 30], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[:, :, 30])
plt.show()

# 定义一个随机仿射变换，输出（300、300、50）图像补丁
rand_affine = RandAffined(
    keys=["image", "label"],
    mode=("bilinear", "nearest"),
    prob=1.0,
    spatial_size=(300, 300, 50),
    translate_range=(40, 40, 2),
    rotate_range=(np.pi / 36, np.pi / 36, np.pi / 4),
    scale_range=(0.15, 0.15, 0.15),
    padding_mode="border",
)
rand_affine.set_random_state(seed=123)