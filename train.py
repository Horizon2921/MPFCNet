from timm.utils import AverageMeter
# from old_init import Options
from init1011 import  Options
from Network.networks import build_net, update_learning_rate, build_UNETR, build_unet, build_vnet, build_swinUNet, build_HRNet, build_hrLKA
import logging
import sys
from glob import glob
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import list_data_collate, decollate_batch, CacheDataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (EnsureType, Compose, LoadImaged, AddChanneld, AsDiscrete, RandGaussianSmoothd,
                              CropForegroundd, SpatialPadd, KeepLargestConnectedComponent,
                              ScaleIntensityd, ToTensord, RandSpatialCropd, Rand3DElasticd, RandAffined, Spacingd,
                              RandShiftIntensityd, RandScaleIntensityd,
                              RandGaussianNoised, RandAdjustContrastd, NormalizeIntensityd, RandFlipd, SaveImage,
                              EnsureChannelFirstd, EnsureTyped, RandCropByLabelClassesd, CenterSpatialCrop,
                              CenterSpatialCropd, RandCropByPosNegLabeld, Orientationd)
from monai.visualize import plot_2d_or_3d_image
from monai.utils import set_determinism
from tqdm import tqdm
import glob
from time import time
import os
from Network.HRNet_att import *
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from init import Options

torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    opt = Options().parse()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = torch.device("cuda")

    train_images = sorted(
        glob.glob(os.path.join(opt.images_folder,  "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(opt.labels_folder,  "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    train_dicts, val_dicts = data_dicts[:-200], data_dicts[-5:]
    print('Number of training images per epoch:', len(train_dicts))
    print('Number of validating images per epoch:', len(val_dicts))


    train_transforms = [
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            # RandCropByLabelClassesd(
            #     keys=["image", "label"],
            #     label_key="label",
            #     image_key="image",
            #     spatial_size=(96,96,96),
            #     num_samples=1,
            #     num_classes=4,
            #     ratios=[1,3,3,3],
            # ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=1),
            RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=2),
        ]

    val_transforms = [
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
        ]

    train_transforms = Compose(train_transforms)
    val_transforms = Compose(val_transforms)

    train_ds = CacheDataset(    # 提前缓存读取
        data=train_dicts, transform=train_transforms, cache_num=12, copy_cache=True,
        cache_rate=1.0, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=list_data_collate, num_workers=4, pin_memory=True)

    val_ds = CacheDataset(
        data=val_dicts, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate, pin_memory=True)


    post_pred = Compose([EnsureType(),  AsDiscrete(argmax=True, to_onehot=6), KeepLargestConnectedComponent(is_onehot=True) ])
    post_label = Compose([EnsureType(),  AsDiscrete(to_onehot=6),KeepLargestConnectedComponent(is_onehot=True)])
    saver_ori = SaveImage(output_dir='/home/dluser/dataset/nnUNetFrame/Pytorch--3D-Medical-Images-Segmentation/result/LKA', output_ext=".nii.gz", output_postfix="ori",print_log=True)
    saver_gt = SaveImage(output_dir='/home/dluser/dataset/nnUNetFrame/Pytorch--3D-Medical-Images-Segmentation/result/LKA', output_ext=".nii.gz", output_postfix="gt",print_log=True)
    saver_seg = SaveImage(output_dir='/home/dluser/dataset/nnUNetFrame/Pytorch--3D-Medical-Images-Segmentation/result/LKA', output_ext=".nii.gz", output_postfix="seg",print_log=True)

    # net = build_HRNet().to(device)
    net = build_hrLKA().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-5)

    ckpt = torch.load(('/home/dluser/dataset/nnUNetFrame/Pytorch--3D-Medical-Images-Segmentation/checkpoint/best_metrich_1024_LKA.pth'))
    optimizer.load_state_dict(ckpt["optim"])
    net.load_state_dict(ckpt["net"])
    # epoch = 0
    # epoch.load_state_dict(ckpt["epoch"])

    # loss
    loss_function = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True  # for accelerating Convs
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)  # Mean Dice caculate
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    scaler = torch.cuda.amp.GradScaler()
    # optimizer


    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    metric_values_tb = []
    metric_values_tc = []
    metric_values_fb = []
    metric_values_lfc = []
    metric_values_rfc = []
    train_time = time()
    writer = SummaryWriter()
    set_determinism(seed=666)


    # net.load_state_dict(torch.load('best_metric_model_test_0824_HRNet.pth'))
    # print("Use pretrained model 824 HRNet ! Attention")

    for epoch in range(opt.epochs):
        print("-" * 20)
        epoch_start = time()
        net.train()
        run_loss = AverageMeter()
        epoch_loss = 0
        pbar = tqdm(train_loader, dynamic_ncols=True)
        for batch_data in pbar:
            pbar.set_description(desc='Train_Epoch : {}/{}'.format(epoch+1, opt.epochs))


            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            # print(inputs[0].shape)
            with torch.cuda.amp.autocast():
                outputs = net(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        print(f"epoch {epoch + 1} average loss: {loss:.4f}  ")
        #     # run_loss.update(loss.item(), n=2)
        # epoch_loss_values.append(run_loss.avg)
        # print(f"epoch {epoch + 1} average loss: {run_loss.avg:.4f}  ")
        if (epoch + 1) % 5 == 0:
            torch.save({'epoch': epoch,
                        'optim': optimizer.state_dict(),
                        'net': net.state_dict()},
                        os.path.join(opt.output_folder, 'latest_checkpoint_1027_LKA.pth'))


        saver_flag = 2
        if (epoch + 1) % val_interval == 0:
            net.eval()

            with torch.no_grad():
                pbar = tqdm(val_loader, ncols=50, dynamic_ncols=True)
                for val_data in pbar:
                # for val_data in val_loader:
                    pbar.set_description(desc='Valid_Epoch ')
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
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)

                    if (saver_flag > 0):
                        meta_data = decollate_batch(val_data["image_meta_dict"])
                        saver_flag = 0
                        print(val_inputs[0].shape, val_outputs[0].shape, val_labels[0].shape)
                        saver_ori(val_inputs[0])
                        saver_seg(torch.argmax(val_outputs[0], dim=0, keepdim=True))
                        saver_gt(torch.argmax(val_labels[0], dim=0, keepdim=True))
                        # saver_ori(val_inputs[0], meta_data[0])
                        # saver_seg(torch.argmax(val_outputs[0], dim=0, keepdim=True), meta_data[0])
                        # saver_gt(torch.argmax(val_labels[0], dim=0, keepdim=True), meta_data[0])
                # aggregate the final mean dice result
                dice = dice_metric.aggregate().item()
                metric_values.append(dice)
                metric_batch = dice_metric_batch.aggregate()
                # print('metric_batch.shape', metric_batch.shape)
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

                # reset the status for next validation round
                dice_metric.reset()
                dice_metric_batch.reset()


                metric_values.append(dice)
                if dice > best_metric:
                    best_metric = dice
                    best_metric_epoch = epoch + 1
                    torch.save(
                        { "net": net.state_dict(),
                          "optim": optimizer.state_dict(),
                          "epoch": epoch},
                        os.path.join(opt.output_folder, 'best_metrich_1027_LKA.pth')
                    )


                print(
                    f"current epoch: {epoch + 1}     mean dice: {dice:.4f}      "
                    # f"average dice:({metric_tb:.4f}, {metric_tc:.4f}, {metric_fb:.4f})"
                    f"average dice:({metric_tb:.4f}, {metric_tc:.4f}, {metric_fb:.4f}, {metric_lfc:.4f}, {metric_rfc:.4f})"
                    f"\nbest mean dice: {best_metric:.4f}    "
                    f"at epoch: {best_metric_epoch}     "
                    f"this epoch tooks {time() - epoch_start:.4f} s "
                )
                writer.add_scalar("Mean_epoch_loss", run_loss.avg, epoch + 1)
                writer.add_scalar("Femoral Bone_dice", metric_tb, epoch + 1)
                writer.add_scalar("Femoral Cartilage_dice", metric_tc, epoch + 1)
                writer.add_scalar("Tibial Bone_dice", metric_fb, epoch + 1)
                writer.add_scalar("Tibial L Cartilage_dice", metric_lfc, epoch + 1)
                writer.add_scalar("Tibial R Cartilage_dice", metric_rfc, epoch + 1)

                plot_2d_or_3d_image(val_inputs, epoch+1, writer, index=0, tag="validation image")
                plot_2d_or_3d_image(val_labels, epoch+1, writer, index=0, tag="validation label")
                plot_2d_or_3d_image(val_outputs, epoch+1, writer, index=0, tag="validation outputs")


                # if opt.network == 'nnunet':  # nnunet需要更新计划
                #     update_learning_rate(net_scheduler, optimizer)

    print(
        "-" * 30,
        f"Train finished ! It took { (time() - train_time)//3600:.4f}hours")
    writer.close()

if __name__ == "__main__":
    main()