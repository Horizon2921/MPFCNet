from init import Options
import monai
from torch.optim import lr_scheduler
from monai.networks.nets import unet
from monai.networks.nets import UNETR, UNet, VNet, SwinUNETR
from Network.HRNet import *
from Network.HRNet_LKA import *
from Network.dsbn import *
from Network.hr_lka_hor import *
from torch.nn import init
import os
import sys
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch + 1 - opt.epochs/2) / float(opt.epochs/2 + 1)
            lr_l = (1 - epoch / opt.epochs) ** 0.9
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    # print('learning rate = %.7f' % lr)



def build_net():

    from init import Options
    opt = Options().parse()

    # create nn-Unet
    if opt.resolution is None:
        sizes, spacings = opt.patch_size, opt.spacing
    else:
        sizes, spacings = opt.patch_size, opt.resolution

    strides, kernels = [], []

    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])

    # # create Unet

    nn_Unet = monai.networks.nets.DynUNet(
        spatial_dims=3,
        in_channels=opt.in_channels,
        out_channels=opt.out_channels,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        res_block=True,
    )

    init_weights(nn_Unet, init_type='normal')

    return nn_Unet


def build_hrLKA():
    net = HighResolutionNetLKA()
    return net

def build_hrLKAspade():
    net = HighResolutionNetLKASPAde()
    return net


def build_unet():
    unet = UNet(spatial_dims=3,
    in_channels=1,
    out_channels=6,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,)

    return unet

def build_swinUNet():
    swinUNETR = SwinUNETR(
        img_size=(96,96,96),
        in_channels=1,
        out_channels=6,
        feature_size=48,
        use_checkpoint=False,
    )
    return swinUNETR

def build_hhh():
    net = HHH()
    return  net

def build_HRNet():
    HRNet = HighResolutionNet()
    return HRNet

def build_vnet():
    vnet = VNet(
        spatial_dims=3,
        in_channels= 1,
        out_channels=6,
        act="elu",
        dropout_prob= 0.5,
        dropout_dim= 3,
        bias=False,

    )

    return vnet

def build_UNETR():

    from init import Options
    opt = Options().parse()

    # create UneTR

    UneTR = UNETR(
        in_channels=1,
        out_channels=6,
        img_size=(96,96,96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )

    init_weights(UneTR, init_type='normal')

    return UneTR


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    # from torchsummaryX import summary
    from torch.nn import init
    from torchstat import stat
    opt = Options().parse()

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # network = build_hrLKA()
    # network = build_UNETR()
    net = build_hrLKA().cpu().eval()
    device = torch.device("cpu")
    # data = torch.randn(1, 1, 96,96,96).to(device)
    # macs . params = profile(net, data)
    #out = net(data)
    # print("out size: {}".format(out.size()))

    # torch.onnx.export(net, data, "Unet_model_graph.onnx")

    summary(net, (1,96,96,96))
    # from ptflops import get_model_complexity_info
    #
    # macs, params = get_model_complexity_info(net, (1, 96, 96, 96), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))




