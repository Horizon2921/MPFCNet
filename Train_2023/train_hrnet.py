from timm.utils import AverageMeter
from init1011 import  Options
from Network.networks import build_net, update_learning_rate, build_UNETR, build_unet, build_vnet, build_swinUNet, build_HRNet
import logging
import sys
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import list_data_collate, decollate_batch, CacheDataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, SurfaceDistanceMetric
from monai.transforms import (EnsureType, Compose, LoadImaged, AsDiscrete, RandGaussianSmoothd,CropForegroundd, SpatialPadd, KeepLargestConnectedComponent,
                              ScaleIntensityd, RandSpatialCropd, Rand3DElasticd, RandAffined, Spacingd, RandShiftIntensityd, RandScaleIntensityd,
                              RandGaussianNoised, RandAdjustContrastd, NormalizeIntensityd, RandFlipd, SaveImage, EnsureChannelFirstd, EnsureTyped, RandCropByLabelClassesd, CenterSpatialCrop,
                              CenterSpatialCropd, RandCropByPosNegLabeld, Orientationd)
from monai.visualize import plot_2d_or_3d_image
from monai.utils import set_determinism
from tqdm import tqdm
import glob
from time import time
import os
from Network.HRNet_att import *
os.environ['CUDA_VISIBLE_DEVICES'] = '1'



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
    train_dicts, val_dicts = data_dicts[:-300], data_dicts[-3:]
    print('Number of training images per epoch:', len(train_dicts))
    print('Number of validating images per epoch:', len(val_dicts))

    # 数据增强方式
    train_transforms = [
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=3,
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

    # 数据读取-dataloader
    train_ds = CacheDataset(    # 提前缓存读取
        data=train_dicts, transform=train_transforms, cache_num=12, copy_cache=True,
        cache_rate=1.0, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=list_data_collate, num_workers=4, pin_memory=True)

    val_ds = CacheDataset(
        data=val_dicts, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate, pin_memory=True)

    # 后处理 post -process
    post_pred = Compose([EnsureType(),  AsDiscrete(argmax=True, to_onehot=6), KeepLargestConnectedComponent(is_onehot=True) ])
    post_label = Compose([EnsureType(),  AsDiscrete(to_onehot=6),KeepLargestConnectedComponent(is_onehot=True)])
    # 在训练过程中保存预测结果
    saver_ori = SaveImage(output_dir='/home/dluser/dataset/nnUNetFrame/Pytorch--3D-Medical-Images-Segmentation/result2023/hrnet', output_ext=".nii.gz", output_postfix="ori",print_log=True)
    saver_gt = SaveImage(output_dir='/home/dluser/dataset/nnUNetFrame/Pytorch--3D-Medical-Images-Segmentation/result2023/hrnet', output_ext=".nii.gz", output_postfix="gt",print_log=True)
    saver_seg = SaveImage(output_dir='/home/dluser/dataset/nnUNetFrame/Pytorch--3D-Medical-Images-Segmentation/result2023/hrnet', output_ext=".nii.gz", output_postfix="seg",print_log=True)

    # 定义模型，优化器，是否选择读取预训练权重（注释掉了）
    net = build_HRNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-5)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.99, weight_decay=3e-5, nesterov=True, )
    # net_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / opt.epochs) ** 0.9)


    # ckpt = torch.load(
    #     '/home/dluser/dataset/nnUNetFrame/Pytorch--3D-Medical-Images-Segmentation/checkpoint/latest_checkpoint_1017_monai_unetr.pth')
    # net.load_state_dict(ckpt["net"])
    # optimizer.load_state_dict(ckpt['model_'])
    # print(f"-" * 20,
    #        f"\nuse best_metric")


    # loss 损失函数
    loss_function = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True)
    torch.backends.cudnn.benchmark = True  # for accelerating Convs
    # 评价指标 Metric
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)  # Mean Dice caculate
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    asd_metric_channel = SurfaceDistanceMetric(include_background=False, reduction='mean_batch', symmetric=True)

    scaler = torch.cuda.amp.GradScaler()


    # 训练的各种参数记录
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


    # 训练epoch
    for epoch in range(300):
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

        # 每隔5个epoch存储一次权重， '' 需要自己定义好名字，读取的时候不能写错
        if (epoch + 1) % 5 == 0:
            torch.save(
                {"net": net.state_dict(),
                 "optim": optimizer.state_dict(),
                 "epoch": epoch},
                os.path.join(opt.output_folder, 'latest_checkpoint_0305_monai_hrnet.pth')
            )


        saver_flag = 2  # 训练过程中对验证集进行有选择的保存，flag代表标志位，每次训练都存储该样本的预测图像，方便可视化
        if (epoch + 1) % val_interval == 0:
            net.eval()
            with torch.no_grad():
                pbar = tqdm(val_loader, ncols=50, dynamic_ncols=True)
                for val_data in pbar:
                # for val_data in val_loader:
                    pbar.set_description(desc='Valid_Epoch')
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
                    asd_metric_channel(y_pred=val_outputs, y=val_labels)

                    if (saver_flag > 0):
                        saver_flag = 0
                        # print(val_inputs[0].shape, val_outputs[0].shape, val_labels[0].shape)
                        print("-" * 10)
                        saver_ori(val_inputs[0])
                        saver_seg(torch.argmax(val_outputs[0], dim=0, keepdim=True))
                        saver_gt(torch.argmax(val_labels[0], dim=0, keepdim=True))
                        # meta_data = decollate_batch(val_data["image_meta_dict"])
                        # saver_ori(val_inputs[0], meta_data[0])
                        # saver_seg(torch.argmax(val_outputs[0], dim=0, keepdim=True), meta_data[0])
                        # saver_gt(torch.argmax(val_labels[0], dim=0, keepdim=True), meta_data[0])

                # aggregate the final mean dice result
                dice = dice_metric.aggregate().item()
                metric_values.append(dice)
                metric_batch = dice_metric_batch.aggregate()
                # print('metric_batch.shape', metric_batch.shape)
                asd_channel = asd_metric_channel.aggregate()
                # print('asd_metric.shape', asd_channel.shape)
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
                asd_metric_channel.reset()

                metric_values.append(dice)
                if dice > best_metric:
                    best_metric = dice
                    best_metric_epoch = epoch + 1
                    torch.save(
                        { "net": net.state_dict(),
                          "optim": optimizer.state_dict(),
                          "epoch": epoch},
                        os.path.join(opt.output_folder, 'best_metric_0305_monai_hrnet.pth')
                    )
                print('')

                print("(no BG) asd: ", asd_channel)
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

    print(
        "-" * 30,
        f"Train finished ! It took { (time() - train_time)//3600:.4f}hours")
    writer.close()

if __name__ == "__main__":
    main()