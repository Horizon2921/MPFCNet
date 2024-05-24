import torch.nn as nn
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torchsummary import summary
BN_MOMENTUM = 0.1
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import os
import sys
import torch.fft
import math
import torch.nn as nn
import torch
from monai.networks.blocks.dynunet_block import UnetOutBlock
from torchsummary import summary
BN_MOMENTUM = 0.1
import traceback

import torch.utils.checkpoint as checkpoint

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
'''
    
'''
if 'DWCONV_IMPL' in os.environ:
        def get_dwconv(dim, kernel, bias):
            return nn.Conv3d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

        # print('[fail to use Megvii Large kernel] Using PyTorch large kernel dw conv impl')
else:
    def get_dwconv(dim, kernel, bias):
        return nn.Conv3d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # print(self.weight.size())
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

# gN Conv
class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv3d(dim, 2 * dim, 1)
        if gflayer is None:  # 这里就是定义上图中的深度可分离卷积
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        # self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        self.proj_out = nn.Conv3d(dim, dim, 1)  # 这里就是第一个映射层
        self.pws = nn.ModuleList(
            [nn.Conv3d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )
        self.scale = s
        # print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f'%self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W, D = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), axis=1)  # 第一个分离部分

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, axis=1)  # 将特征分成对应的几个部分，也就是第二个split
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]  # 这里就是逐元素相乘操作

        x = self.proj_out(x)

        return x

class gconvBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)  # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W, D = x.shape # 0 1 2 3 4

        x = x + self.drop_path(1 * self.gnconv(self.norm1(x)))

        input = x
        # x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        # x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, D, C) -> (N, C, H, W， D)
                                   #  0  1  2  3  4
        x = input + self.drop_path(x)
        return x

class gnconvBlock(nn.Module):
    r""" HorNet block
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = nn.BatchNorm3d(128, momentum=BN_MOMENTUM)
        self.gnconv = gnconv(dim)  # depthwise conv
        self.norm2 = nn.BatchNorm3d(128, momentum=BN_MOMENTUM)
        self.pwconv1 = nn.Linear(dim, dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear( dim, dim // 4)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W, D = x.shape # 0 1 2 3 4

        x = x + self.drop_path(1 * self.gnconv(self.norm1(x)))

        input = x
        # x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        # x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, D, C) -> (N, C, H, W， D)
                                   #  0  1  2  3  4
        x = input + self.drop_path(x)
        return x

# HamBurger
class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.spatial = getattr(args, 'SPATIAL', True)

        self.S = getattr(args, 'MD_S', 1)
        self.D = getattr(args, 'MD_D', 512)
        self.R = getattr(args, 'MD_R', 64)

        self.train_steps = getattr(args, 'TRAIN_STEPS', 6)
        self.eval_steps  = getattr(args, 'EVAL_STEPS', 7)

        self.inv_t = getattr(args, 'INV_T', 100)
        self.eta   = getattr(args, 'ETA', 0.9)

        self.rand_init = getattr(args, 'RAND_INIT', True)

        print('spatial', self.spatial)
        print('S', self.S)
        print('D', self.D)
        print('R', self.R)
        print('train_steps', self.train_steps)
        print('eval_steps', self.eval_steps)
        print('inv_t', self.inv_t)
        print('eta', self.eta)
        print('rand_init', self.rand_init)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        if not self.rand_init and not self.training and not return_bases:
            self.online_update(bases)

        # if not self.rand_init or return_bases:
        #     return x, bases
        # else:
        return x

    @torch.no_grad()
    def online_update(self, bases):
        # (B, S, D, R) -> (S, D, R)
        update = bases.mean(dim=0)
        self.bases += self.eta * (update - self.bases)
        self.bases = F.normalize(self.bases, dim=1)

class HamburgerV2(nn.Module):
    def __init__(self, in_c, args=None):
        super().__init__()

        ham_type = getattr(args, 'HAM_TYPE', 'VQ')

        C = getattr(args, 'MD_D', 512)

        if ham_type == 'NMF':
            self.lower_bread = nn.Sequential(nn.Conv2d(in_c, C, 1),
                                             nn.ReLU(inplace=True))
        else:
            self.lower_bread = nn.Conv2d(in_c, C, 1)

        HAM = get_hams(ham_type)
        self.ham = HAM(args)

        self.cheese = ConvBNReLU(C, C)
        self.upper_bread = nn.Conv2d(C, in_c, 1, bias=False)

        self.shortcut = nn.Sequential()

        self._init_weight()

        print('ham', HAM)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                N = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / N))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.lower_bread(x)
        x = self.ham(x)
        x = self.cheese(x)
        x = self.upper_bread(x)

        x = F.relu(x + shortcut, inplace=True)

        return x

    def online_update(self, bases):
        if hasattr(self.ham, 'online_update'):
            self.ham.online_update(bases)


''' VAN-Net'''
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv3d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn
class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
    
class VANBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm3d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


'''
    [conv -> bn -> relu -> conv -> bn -> Residual -> relu  
'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class Multi_channel_Block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

        self.conv3 = nn.Conv3d(inplanes, planes, kernel_size=1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

'''
    2x[conv -> bn -> relu] ->  Residual -> relu  
'''
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList() # 存储每一个branch上的block
        for i in range(self.input_branches):  # 每个分支上都先通过不同个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数，每一层的通道数要翻倍
            branch = nn.Sequential(
                BasicBlock(w, w),
                gnconvBlock(w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            self.branches.append(branch)  # 每一个分支上的Block已构建好

        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv3d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm3d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='trilinear', align_corners=True)
                        )
                    )
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv3d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm3d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样
                    ops.append(
                        nn.Sequential(
                            nn.Conv3d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm3d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 接着融合不同尺寸信息
        x_fused = []
        for i in range(len(self.fuse_layers)):
            x_fused.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))]) # 第j个输出分支对 前面不同分支的输出进行处理，包括不处理（Indenty） 上采样x2 、 x4 ,相加
                )
            )

        return x_fused


class HighResolutionNet(nn.Module):
    def __init__(self, base_channel: int = 32, output_channels: int = 6):
        super().__init__()
        '''
        Stem层， 初始图像带步长卷积下采样了两次，变成1/4尺寸的特征图和c=64）  
        然后进入Layer1. input: 1/4的尺寸+base channel * 4的通道。 只调整channel数
                       有两个分支，分支1再变为base channel /2 ， 分支2变为 1/2尺寸+  base channel
        '''
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(32, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(32, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        dims = [base_channel, base_channel * 2, base_channel * 4, base_channel * 8]
        depths = [3, 3, 9, 3]
        # Stage1
        downsample = nn.Sequential(
            nn.Conv3d(32, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(128, momentum=BN_MOMENTUM)
        )



        '''
        Layer1 在不同的stage 一直卷
        '''
        self.layer1 = nn.Sequential(
            Bottleneck(32, 32, downsample=downsample), #ResNet bottleneck 操作，输入为c，输出为4c
            Bottleneck(128, 32),
            Bottleneck(128, 32),
            Bottleneck(128, 32)
        )

        self.transition1 = nn.ModuleList([  # 两个分支，1/4尺寸和1/8尺寸+
            nn.Sequential(
                nn.Conv3d(128, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(base_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(  # 这里又使用一次Sequential是为了适配原项目中提供的权重
                    nn.Conv3d(128, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(base_channel * 2, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage2
        self.stage2 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel)
        )

        # transition2 ,先对Stage2输出的两个Block不做处理，下采样第二个Block
        self.transition2 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv3d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(base_channel * 4, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage3
        self.stage3 = nn.Sequential(
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel)
        )

        # transition3
        self.transition3 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv3d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(base_channel * 8, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage4
        # 注意，最后一个StageModule只输出分辨率最高的特征层
        self.stage4 = nn.Sequential(
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=1, c=base_channel)
        )

        self.attn = VANBlock(base_channel)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        # Final layer
        self.final_layer = nn.Conv3d(base_channel*2, output_channels, kernel_size=1, stride=1)

        self.out = UnetOutBlock(spatial_dims=1, in_channels=48, out_channels=output_channels)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        residual = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)  # stem

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # x变成了一个列表。每个Stage有好几个输出

        x = self.stage2(x)  # 把前一层的x输入传入
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]),
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x) # 4个输入分支，1个输出分支
        # stage4输出为1/2大小，需要上采用和Stem做Concat
        x = x[0]
        x = self.attn(x)
        x = self.up(x)

        x = self.final_layer(torch.cat((x, residual),dim=1))


        #print('x shape', x.shape)

        return x

if __name__ == '__main__':
    torch.cuda.set_device(0)
    network = HighResolutionNet()
    net = network.cuda().eval()

    summary(net,(1,96,96,96))
