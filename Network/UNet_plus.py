import torch
from torch import nn
from torchsummary import summary

class Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels,):
        super.__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(middle_channels)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)


        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out

class VGGBlock(nn.Module):
    '''
        define a block of operations: 2x[Conv -> BN -> Relu]
        定义一组操作。2x[Conv -> BN -> Relu]
    '''
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(middle_channels)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet_plus(nn.Module):
    def __init__(self, num_classes, input_channels, **kwargs):
        super().__init__()
        '''
            nb_filter定义每组block操作输出的channel
            pool 和 up 已知
            forward时 torch.cat在channel维度拼接
        '''
        nb_filter = [32, 64, 128, 256, 320]

        self.pool = nn.MaxPool3d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = Block(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = Block(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = Block(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = Block(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        print('shape of x3_0', x3_0.shape)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], dim=1))
        print('shape of x3_1', x3_1.shape)

        output = self.final(x0_4)
        return output

UNet_plus = UNet_plus

if __name__ == '__main__':
    torch.cuda.set_device(1)
    network = UNet_plus(num_classes=6, input_channels=1)

    net = network.cuda().eval()

    summary(net,(1,96,96,96))