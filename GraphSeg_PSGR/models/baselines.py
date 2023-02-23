import torch
import torch.nn as nn
from torch.nn import init
from models.sync_batchnorm import SynchronizedBatchNorm2d
from utilis import criterion
import torch.nn.functional as F


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding, BatchNorm):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            BatchNorm(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            BatchNorm(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            BatchNorm(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, BatchNorm):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = BatchNorm(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = BatchNorm(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, BatchNorm):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, BatchNorm):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, BatchNorm, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, BatchNorm, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, BatchNorm, t=t),
            Recurrent_block(ch_out, BatchNorm, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out, BatchNorm):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            BatchNorm(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int, BatchNorm):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            BatchNorm(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            BatchNorm(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            BatchNorm(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        diffY = x1.size()[2] - g1.size()[2]
        diffX = x1.size()[3] - g1.size()[3]

        g1 = F.pad(g1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class R2U_Net(nn.Module):
    def __init__(self, n_channels, n_classes, sync_bn=True, t=1):
        super(R2U_Net, self).__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(n_channels, 64, BatchNorm, t=t)

        self.RRCNN2 = RRCNN_block(64, 128, BatchNorm, t=t)

        self.RRCNN3 = RRCNN_block(128, 256, BatchNorm, t=t)

        self.RRCNN4 = RRCNN_block(256, 512, BatchNorm, t=t)

        self.RRCNN5 = RRCNN_block(512, 1024, BatchNorm, t=t)

        self.Up5 = up_conv(1024, 512, BatchNorm)
        self.Up_RRCNN5 = RRCNN_block(1024, 512, BatchNorm, t=t)

        self.Up4 = up_conv(512, 256, BatchNorm)
        self.Up_RRCNN4 = RRCNN_block(512, 256, BatchNorm, t=t)

        self.Up3 = up_conv(256, 128, BatchNorm)
        self.Up_RRCNN3 = RRCNN_block(256, 128, BatchNorm, t=t)

        self.Up2 = up_conv(128, 64, BatchNorm)
        self.Up_RRCNN2 = RRCNN_block(128, 64, BatchNorm, t=t)

        self.Conv_1x1 = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)

        self.criterion = getattr(criterion, 'bce_gdl')


    def forward(self, x, gt=None):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        loss = self.criterion(d1, gt)

        return loss, d1


class AttU_Net(nn.Module):
    def __init__(self, n_channels, n_classes, sync_bn=True):
        super(AttU_Net, self).__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(n_channels, 64, BatchNorm)
        self.Conv2 = conv_block(64, 128, BatchNorm)
        self.Conv3 = conv_block(128, 256, BatchNorm)
        self.Conv4 = conv_block(256, 512, BatchNorm)
        self.Conv5 = conv_block(512, 1024, BatchNorm)

        self.Up5 = up_conv(1024, 512, BatchNorm)
        self.Att5 = Attention_block(512, 512, 256, BatchNorm)
        self.Up_conv5 = conv_block(1024, 512, BatchNorm)

        self.Up4 = up_conv(512, 256, BatchNorm)
        self.Att4 = Attention_block(256, 256, 128, BatchNorm)
        self.Up_conv4 = conv_block(512, 256, BatchNorm)

        self.Up3 = up_conv(256, 128, BatchNorm)
        self.Att3 = Attention_block(128, 128, 64, BatchNorm)
        self.Up_conv3 = conv_block(256, 128, BatchNorm)

        self.Up2 = up_conv(128, 64, BatchNorm)
        self.Att2 = Attention_block(64, 64, 32, BatchNorm)
        self.Up_conv2 = conv_block(128, 64, BatchNorm)

        self.Conv_1x1 = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)

        self.criterion = getattr(criterion, 'bce_gdl')


    def forward(self, x, gt=None):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        loss = self.criterion(d1, gt)

        return loss, d1


class R2AttU_Net(nn.Module):
    def __init__(self, n_channels, n_classes, sync_bn=True, t=1):
        super(R2AttU_Net, self).__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(n_channels, 64, BatchNorm, t=t)

        self.RRCNN2 = RRCNN_block(64, 128, BatchNorm, t=t)

        self.RRCNN3 = RRCNN_block(128, 256, BatchNorm, t=t)

        self.RRCNN4 = RRCNN_block(256, 512, BatchNorm, t=t)

        self.RRCNN5 = RRCNN_block(512, 1024, BatchNorm, t=t)

        self.Up5 = up_conv(1024, 512, BatchNorm)
        self.Att5 = Attention_block(512, 512, 256, BatchNorm)
        self.Up_RRCNN5 = RRCNN_block(1024, 512, BatchNorm, t=t)

        self.Up4 = up_conv(512, 256, BatchNorm)
        self.Att4 = Attention_block(256, 256, 128, BatchNorm)
        self.Up_RRCNN4 = RRCNN_block(512, 256, BatchNorm, t=t)

        self.Up3 = up_conv(256, 128, BatchNorm)
        self.Att3 = Attention_block(128, 128, 64, BatchNorm)
        self.Up_RRCNN3 = RRCNN_block(256, 128, BatchNorm, t=t)

        self.Up2 = up_conv(128, 64, BatchNorm)
        self.Att2 = Attention_block(64, 64, 32, BatchNorm)
        self.Up_RRCNN2 = RRCNN_block(128, 64, BatchNorm, t=t)

        self.Conv_1x1 = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)

        self.criterion = getattr(criterion, 'bce_gdl')


    def forward(self, x, gt=None):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)

        diffY = x1.size()[2] - d2.size()[2]
        diffX = x1.size()[3] - d2.size()[3]

        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        loss = self.criterion(d1, gt)

        return loss, d1


class NestedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, sync_bn=True, deep_supervision=False):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(n_channels, nb_filter[0], nb_filter[0], BatchNorm)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], BatchNorm)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], BatchNorm)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], BatchNorm)
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4], BatchNorm)

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0], BatchNorm)
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1], BatchNorm)
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2], BatchNorm)
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3], BatchNorm)

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0], BatchNorm)
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1], BatchNorm)
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2], BatchNorm)

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0], BatchNorm)
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1], BatchNorm)

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0], BatchNorm)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)

        self.criterion = getattr(criterion, 'bce_gdl')

    def forward(self, input, gt=None):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            loss = self.criterion(output, gt)
            return loss, output


class ResUnet(nn.Module):
    def __init__(self, n_channels, n_classes, sync_bn=True, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.input_layer = nn.Sequential(
            nn.Conv2d(n_channels, filters[0], kernel_size=3, padding=1),
            BatchNorm(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(n_channels, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1, BatchNorm)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1, BatchNorm)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1, BatchNorm)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1, BatchNorm)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1, BatchNorm)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1, BatchNorm)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], n_classes, 1, 1),
        )

        self.criterion = getattr(criterion, 'bce_gdl')

    def forward(self, x, gt=None):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        loss = self.criterion(output, gt)

        return loss, output