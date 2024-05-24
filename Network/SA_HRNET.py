import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4)
#        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x = x.view(-1, self.channels, x.shape[2] * x.shape[3] * x.shape[4]).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).reshape(-1, self.channels, self.size, self.size, self.size)


#from torchsummary import summary
BN_MOMENTUM = 0.1

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
                BasicBlock(w, w),
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

# def get_b16_config():
#     """Returns the ViT-B/16 configuration."""
#     config = ml_collections.ConfigDict()
#     config.patches = ml_collections.ConfigDict({'size': (14, 14)})
#     config.hidden_size = 168
#     config.transformer = ml_collections.ConfigDict()
#     config.transformer.mlp_dim = 3072
#     config.transformer.num_heads = 12
#     config.transformer.num_layers = 12
#     config.transformer.attention_dropout_rate = 0.0
#     config.transformer.dropout_rate = 0.1
#
#     config.classifier = 'seg'
#     config.representation_size = None
#     config.resnet_pretrained_path = None
#     config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
#     config.patch_size = 16
#
#     config.decoder_channels = (256, 128, 64, 16)
#     config.n_classes = 2
#     config.activation = 'softmax'
#     return config
#
#
# def get_testing():
#     """Returns a minimal configuration for testing."""
#     config = ml_collections.ConfigDict()
#     config.patches = ml_collections.ConfigDict({'size': (16, 16)})
#     config.hidden_size = 1
#     config.transformer = ml_collections.ConfigDict()
#     config.transformer.mlp_dim = 1
#     config.transformer.num_heads = 1
#     config.transformer.num_layers = 1
#     config.transformer.attention_dropout_rate = 0.0
#     config.transformer.dropout_rate = 0.1
#     config.classifier = 'token'
#     config.representation_size = None
#     return config
# import ml_collections
# def get_r50_b16_config():
#     """Returns the Resnet50 + ViT-B/16 configuration."""
#     config = get_b16_config()
#     config.patches.grid = (14, 14)
#     config.resnet = ml_collections.ConfigDict()
#     config.resnet.num_layers = (3, 4, 9)
#     config.resnet.width_factor = 1
#
#     config.classifier = 'seg'
#     config.pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
#     config.decoder_channels = (256, 128, 64, 16)
#     config.skip_channels = [512, 256, 64, 16]
#     config.n_classes = 2
#     config.n_skip = 3
#     config.activation = 'softmax'
#
#     return config

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        self.patch_embeddings = nn.Conv3d(in_channels=256,
                                       out_channels=168,
                                       kernel_size=1,
                                       stride=1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, 245, 168))
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class HighResolutionNet(nn.Module):
    def __init__(self, base_channel: int = 32, output_channels: int = 4, time_dim=256, device="cpu",d_model=128, num_classes=6):
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
        self.sa1 = SelfAttention(128, 24)
        self.time_dim = time_dim
        self.device = device
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)
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

        self.sa2 = SelfAttention(32, 96)

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

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        # Final layer
        self.final_layer = nn.Conv3d(base_channel*2, output_channels, kernel_size=1, stride=1)




    # def forward(self, x, t ,y ):
    #     t = t.unsqueeze(-1).type(torch.float)
    #     t = self.pos_encoding(t, self.time_dim)
    #
    #     if y is not None:
    #         t += self.label_emb(y)
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        residual = x
        print(residual.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)  # stem
        # x = self.sa1(x)

        x = self.layer1(x)
        # （2，32，48，48，48）
        # （2，128，48，48，48）
        x = [trans(x) for trans in self.transition1]  # x变成了一个列表。每个Stage有好几个输出
        # （16，128，24，24，24）

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
        x = self.up(x)  #(2,32,48,48,48)
        print(x)

        x = self.sa2(x) #(2,32,96,96,96)
        print("x_sa", x)
        # x = self.final_layer(x) #(1,32,96,96,96)
        x = self.final_layer(torch.cat((x, residual),dim=1))
        print('x shape', x.shape)

        return x
from torchsummary import summary
if __name__ == '__main__':

    # torch.cuda.set_device(0)
    network = HighResolutionNet()
    # net = network.cpu().eval()
    # torch.cuda.set_device(0)

    x = torch.randn(1, 1, 96, 96, 96)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    net = network.cpu().eval()
    # summary(net(x, t, y),(1,96,96,96), device='cpu')
    #print(net(x, t, y).shape)
    # print(net)
    summary(net(x, t, y),(1,96,96,96), device='cpu')