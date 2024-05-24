import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils import *
from Network.networks import build_net, build_swinUNet
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import (Compose, LoadImaged, AddChanneld, CropForegroundd, SaveImaged, EnsureTyped, Invertd, EnsureChannelFirstd)
import glob
from monai.handlers.utils import from_engine


root_dir = '/home/dluser/dataset/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task55_k/'
model_dir = '/home/dluser/dataset/nnUNetFrame/Pytorch--3D-Medical-Images-Segmentation'
test_images = sorted(glob.glob(os.path.join(root_dir, "imagesTs", "*.nii.gz")))

val_files = [{"image": image} for image in test_images]

val_transforms = [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            # CropForegroundd(keys=['image', 'label'], source_key='image'),               # crop CropForeground
            # Spacingd(
            #     keys=["image", "label"],
            #     pixdim=(1.5, 1.5, 2.0),
            #     mode=("bilinear", "nearest"),
            # ),
            RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=1),
            RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=2),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 36, np.pi / 36, np.pi * 2), padding_mode="zeros"),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 36, np.pi / 2, np.pi / 36), padding_mode="zeros"),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 2, np.pi / 36, np.pi / 36), padding_mode="zeros"),
            Rand3DElasticd(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                           sigma_range=(5, 8), magnitude_range=(100, 200), scale_range=(0.15, 0.15, 0.15),
                           padding_mode="zeros"),
            RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15), prob=0.1,),
            RandAdjustContrastd(keys=['image'], gamma=(0.5, 2.5), prob=0.1),
            RandGaussianNoised(keys=['image'], prob=0.1, mean=np.random.uniform(0, 0.5), std=np.random.uniform(0, 1)),
            ToTensord(keys=['image', 'label'])
        ]

val_transforms = Compose(val_transforms)
val_org_ds = Dataset(
    data=val_files, transform=val_transforms)
val_org_loader = DataLoader(val_org_ds, batch_size=1, num_workers=4)

post_transforms = Compose(
    Invertd(
        keys="pred",
        transform=val_transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
    ),
    AsDiscreted(keys="pred", argmax=True, to_onehot=6),
    AsDiscreted(keys="label", to_onehot=6),
)



#Network = build_UNETR()
model = build_swinUNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)  # Mean Dice caculate
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

model.load_state_dict(torch.load(
    os.path.join(model_dir, "best_metric_model_test_0818_swinunet.pth")))

model.eval()
with torch.no_grad():
    for val_data in val_org_loader:
        val_inputs = val_data["image"].to(device)
        roi_size = (96,96,96)
        sw_batch_size = 4
        val_data["pred"] = sliding_window_inference(
            val_inputs, roi_size, sw_batch_size, model)
        val_data = [post_transforms(i) for i in decollate_batch(val_data)]
        val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
        # compute metric for current iteration
        dice_metric(y_pred=val_outputs, y=val_labels)

    # aggregate the final mean dice result
    metric_org = dice_metric.aggregate().item()
    # reset the status for next validation round
    dice_metric.reset()

print("Metric on original image spacing: ", metric_org)
