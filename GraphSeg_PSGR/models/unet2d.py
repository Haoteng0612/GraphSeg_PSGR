import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.self_attention_gcn import SPGR_Unit
from models.modules.non_local import NonLocal
from models.sync_batchnorm import SynchronizedBatchNorm2d
from utilis import criterion
from utilis.utilis import calculate_uncertainty


class GCN(nn.Module):

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.relu(h)
        h = self.conv2(h)
        return h


class GloRe_Unit(nn.Module):
    def __init__(self, num_in, num_mid, BatchNorm, kernel_size=1, initial=False):
        super(GloRe_Unit, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        padding = 1 if kernel_size == 3 else 0

        # reduce dimension
        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate graph transformation function
        self.conv_proj = nn.Conv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)
        # ----------
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # tail: extend dimension
        self.fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=1,
                              groups=1, bias=False)

        self.bn = BatchNorm(num_in)

        if initial:
            self._init_weight()

    def forward(self, x):
        '''
        :param x: (n, c, h, w)
        '''
        batch_size = x.size(0)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(batch_size, self.num_n, -1)
        # x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=2)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: pixel space -> instance space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: instance space -> pixel space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(batch_size, self.num_s, *x.size()[2:])

        # -----------------
        # final
        out = x + self.bn(self.fc_2(x_state))

        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, BatchNorm, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            BatchNorm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, BatchNorm):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, BatchNorm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, BatchNorm, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, BatchNorm, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, BatchNorm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, BatchNorm, kernel_size, stride, padding, dilation):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, sync_bn=True, embedded_module='spgr', gcn=False, np_ratio=0.4,
                 k_ratio=0.5, coarse_seg=True, base_channel=64, os=8):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.np_ratio = np_ratio
        self.embedded_module = embedded_module
        self.gcn = gcn
        self.coarse_seg = coarse_seg
        self.os = os

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        if self.embedded_module == 'spgr':
            self.self_att_gcn = SPGR_Unit(in_channels=base_channel * (os // 2), BatchNorm=BatchNorm, inter_channels=256,
                                          gcn=self.gcn, k_ratio=k_ratio, initial=False)
        elif self.embedded_module == 'non-local':
            self.non_local = NonLocal(in_channels=base_channel * (os // 2), BatchNorm=BatchNorm, inter_channels=256,
                                      initial=False)
        elif self.embedded_module == 'glore':
            self.global_reasoning = GloRe_Unit(num_in=base_channel * (os // 2), num_mid=64, BatchNorm=BatchNorm,
                                               initial=False)

        if self.coarse_seg:
            self.coarse_logits = nn.Sequential(ConvBNReLU(base_channel * (os // 2),
                                                          base_channel * 2, BatchNorm, 3, 1, 1, 1),
                                               nn.Conv2d(base_channel * 2, n_classes, kernel_size=1, stride=1))

        self.inc = DoubleConv(n_channels, base_channel, BatchNorm)  # 1
        self.down1 = Down(base_channel, base_channel * 2, BatchNorm)  # 1/2
        self.down2 = Down(base_channel * 2, base_channel * 4, BatchNorm)  # 1/4
        self.down3 = Down(base_channel * 4, base_channel * 8, BatchNorm)  # 1/8
        self.down4 = Down(base_channel * 8, base_channel * 8, BatchNorm)  # 1/16

        self.up1 = Up(base_channel * 16, base_channel * 4, BatchNorm)  # 1/8
        self.up2 = Up(base_channel * 8, base_channel * 2, BatchNorm)  # 1/4
        self.up3 = Up(base_channel * 4, base_channel, BatchNorm)  # 1/2
        self.up4 = Up(base_channel * 2, base_channel, BatchNorm)  # 1
        self.outc = OutConv(base_channel, n_classes)

        self.criterion = getattr(criterion, 'bce_gdl')

    def forward(self, x, gt=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.os == 16 and self.coarse_seg:
            coarse_logits = self.coarse_logits(x5)
            b, h, w = coarse_logits.shape[0], coarse_logits.shape[2], coarse_logits.shape[3]

            if self.embedded_module == 'spgr':
                with torch.no_grad():
                    K = int(self.np_ratio * h * w)
                    # print('uncertainty nodes num:', K)
                    hp_map = torch.zeros(b * h * w, 1, dtype=torch.long, device=coarse_logits.device)
                    uncertainty_score = calculate_uncertainty(coarse_logits, self.n_classes).view(b, 1, -1)[:, 0,
                                        :]  # b, 1, h, w -> b, hw
                    idx = torch.topk(uncertainty_score, k=K, dim=1)[1]  # b, k
                    shift = (h * w) * torch.arange(b, dtype=torch.long, device=coarse_logits.device)
                    idx += shift[:, None]
                    hp_map[idx.view(-1), :] = 1
                    hp_map = hp_map.view(b, -1)  # b, n

                x5 = self.self_att_gcn(x5, hp_map)

            elif self.embedded_module == 'non-local':
                x5 = self.non_local(x5)

            elif self.embedded_module == 'glore':
                x5 = self.global_reasoning(x5)

        x = self.up1(x5, x4)

        if self.os == 8 and self.coarse_seg:
            coarse_logits = self.coarse_logits(x)
            b, h, w = coarse_logits.shape[0], coarse_logits.shape[2], coarse_logits.shape[3]

            if self.embedded_module == 'spgr':
                with torch.no_grad():
                    K = int(self.np_ratio * h * w)
                    # print('uncertainty nodes num:', K)
                    hp_map = torch.zeros(b * h * w, 1, dtype=torch.long, device=coarse_logits.device)
                    uncertainty_score = calculate_uncertainty(coarse_logits, self.n_classes).view(b, 1, -1)[:, 0,
                                        :]  # b, 1, h, w -> b, hw
                    idx = torch.topk(uncertainty_score, k=K, dim=1)[1]  # b, k
                    shift = (h * w) * torch.arange(b, dtype=torch.long, device=coarse_logits.device)
                    idx += shift[:, None]
                    hp_map[idx.view(-1), :] = 1
                    hp_map = hp_map.view(b, -1)  # b, n

                x = self.self_att_gcn(x, hp_map)

            elif self.embedded_module == 'non-local':
                x = self.non_local(x)

            elif self.embedded_module == 'glore':
                x = self.global_reasoning(x)

        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        if self.coarse_seg:
            coarse_logits = F.interpolate(coarse_logits, size=x.size()[2:], mode='bilinear', align_corners=True)
            loss = self.criterion(x, gt) + self.criterion(coarse_logits, gt) * 0.1
            return loss, x
        else:
            loss = self.criterion(x, gt)
            return loss, x
