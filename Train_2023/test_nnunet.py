from Network.networks import build_net, update_learning_rate, build_UNETR, build_unet, build_vnet, build_swinUNet, build_HRNet
import logging
import sys
import glob
import torch
from torch.utils.data import DataLoader
from monai.data import list_data_collate, decollate_batch, CacheDataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, SurfaceDistanceMetric
from monai.transforms import (EnsureType, Compose, LoadImaged, AddChanneld, AsDiscrete, RandGaussianSmoothd,
                              CropForegroundd, SpatialPadd, ScaleIntensityd, ToTensord, RandSpatialCropd,
                              Rand3DElasticd, RandAffined, RandShiftIntensityd, RandGaussianNoised, RandAdjustContrastd,
                              NormalizeIntensityd, RandFlipd, SaveImage, EnsureChannelFirstd, KeepLargestConnectedComponent)
from monai.visualize import plot_2d_or_3d_image
from tqdm import tqdm
from time import time
import os
from init1011 import  Options
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定GPU1


def main():
    opt = Options().parse()
    # monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train_images = sorted(
        glob.glob(os.path.join(opt.images_folder,  "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(opt.labels_folder,  "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    val_dicts = data_dicts[10:20]
    print('Number of validating images per epoch:', len(val_dicts))


    # Creation of data directories for data_loader
    # Transforms for training and validation

    val_transforms = [
        LoadImaged(keys=['image', 'label']),
        EnsureChannelFirstd(keys=['image', 'label']),
        ]
    val_transforms = Compose(val_transforms)

    # Define CacheDataset and DataLoader for training and validation
    # create a training data loader

    val_ds = CacheDataset(
        data=val_dicts, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=opt.workers, collate_fn=list_data_collate, pin_memory=True)
    post_pred = Compose([EnsureType(),  AsDiscrete(argmax=True, to_onehot=6), KeepLargestConnectedComponent(is_onehot=True) ])
    post_label = Compose([EnsureType(),  AsDiscrete(to_onehot=6), KeepLargestConnectedComponent(is_onehot=True)])

    # 指定输出路径
    output_folder = '/home/dluser/dataset/nnUNetFrame/Pytorch--3D-Medical-Images-Segmentation/result2023/test/nnunet'
    saver_ori = SaveImage(output_dir=output_folder, output_ext=".nii.gz", output_postfix="ori",print_log=True)
    saver_gt = SaveImage(output_dir=output_folder, output_ext=".nii.gz", output_postfix="gt",print_log=True)
    saver_seg = SaveImage(output_dir=output_folder, output_ext=".nii.gz", output_postfix="seg",print_log=True)


    # Create Model, Loss, Optimizer
    device = torch.device("cuda")
    net = build_net()  # nn build_net
    net.to(device)

    torch.backends.cudnn.benchmark = opt.benchmark  # for accelerating Convs
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)  # Mean Dice caculate
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    asd_metric_channel = SurfaceDistanceMetric(include_background=False, reduction='mean_batch', symmetric=True)


    metric_values = []
    metric_values_tb = []
    metric_values_tc = []
    metric_values_fb = []
    metric_values_lfc = []
    metric_values_rfc = []

    val_time = time()

    # net.load_state_dict(torch.load(
    #     '/home/dluser/dataset/nnUNetFrame/Pytorch--3D-Medical-Images-Segmentation/checkpoint/best_metric_model_test_0822_HRNet.pth'))
    # print("Use pretrained model ! Attention")
    ckpt = torch.load(
        '/home/dluser/dataset/nnUNetFrame/Pytorch--3D-Medical-Images-Segmentation/result2023/checkpoint/best_metric_0305_monai_nnunet.pth'
    )
    net.load_state_dict(ckpt["net"])

    net.eval()
    saver_flag = 1
    with torch.no_grad():
            pbar = tqdm(val_loader, ncols=50, dynamic_ncols=True)
            for val_data in pbar:
                # for val_data in val_loader:
                pbar.set_description(desc='Test')
                val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                )

                roi_size = opt.patch_size
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, net)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                asd_metric_channel(y_pred=val_outputs, y=val_labels)
                    # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

                # if (saver_flag > 0):
                #
                #         saver_flag = 0
                        # print(val_inputs[0].shape, val_outputs[0].shape, val_labels[0].shape)
                saver_ori(val_inputs[0])
                saver_seg(torch.argmax(val_outputs[0], dim=0, keepdim=True))
                saver_gt(torch.argmax(val_labels[0], dim=0, keepdim=True))


                # aggregate the final mean dice result
                dice = dice_metric.aggregate().item()
                asd_channel = asd_metric_channel.aggregate()

                metric_values.append(dice)
                metric_batch = dice_metric_batch.aggregate()
                # print('metric_batch.shape', metric_batch.shape)
                # print('asd_channel.shape', asd_channel.shape)
                metric_tb = metric_batch[1].item()
                metric_values_tb.append(metric_tb)
                metric_tc = metric_batch[2].item()
                metric_values_tc.append(metric_tc)
                metric_fb = metric_batch[3].item()
                metric_values_fb.append(metric_fb)
                metric_lfc = metric_batch[4].item()
                metric_values_lfc.append(metric_lfc)
                metric_rfc = metric_batch[5].item()
                metric_values_rfc.append(metric_rfc)
                print(f"\n(no BG)  ASD: ", asd_channel)
                print(
                    f"average dice:({metric_tb:.4f}, {metric_tc:.4f}, {metric_fb:.4f}, {metric_lfc:.4f}, {metric_rfc:.4f})"
                )
                # reset the status for next validation round
                dice_metric.reset()
                dice_metric_batch.reset()
                asd_metric_channel.reset()



    val_end = time()
    print('val takes:{:.4f}'.format(val_end - val_time))


if __name__ == "__main__":
    main()