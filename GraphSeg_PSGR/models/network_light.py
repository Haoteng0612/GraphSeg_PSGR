import torch
import torch.nn as nn
from models.modules.self_attention_gcn import NonLocalGCN
from models.backbone import resnet, res2net
import torch.nn.functional as F
from utilis import criterion
from models.sync_batchnorm import SynchronizedBatchNorm2d
from utilis.utilis import calculate_uncertainty


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, initial=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        if initial:
            self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, SynchronizedBatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def convert_to_separable_conv(module, initial=True):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias, initial=initial)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm, initial=True):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv2 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn2 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        if initial:
            self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        del x1, x2, x3, x4, x5

        x = self.relu(self.bn2(self.conv2(x)))

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, initial=True):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn' or backbone == 'res2net':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        if initial:
            self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.relu(self.bn1(self.conv1(low_level_feat)))

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        del low_level_feat
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CoarseSeg(nn.Module):
    def __init__(self, num_classes, BatchNorm, initial=True):
        super(CoarseSeg, self).__init__()

        self.last_conv = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(128, num_classes, kernel_size=1, stride=1))
        if initial:
            self._init_weight()

    def forward(self, x):

        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class GCNSegNet(nn.Module):
    def __init__(self, num_classes, backbone='resnet', sync_bn=True, pretrained=True, os=16,
                 use_gcn=True, coarse_seg=True, gcn='gcn', np_ratio=0.4, k_ratio=0.5, initial='aspp', sp_conv=True):
        super(GCNSegNet, self).__init__()
        self.np_ratio = np_ratio
        self.pretrained = pretrained
        self.use_gcn = use_gcn
        self.gcn = gcn
        self.coarse_seg = coarse_seg
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        if initial == 'aspp':
            aspp_ini, decoder_ini, coarse_ini, gcn_ini = True, False, False, False
        elif initial == 'all':
            aspp_ini, decoder_ini, coarse_ini, gcn_ini = True, True, True, True
        elif initial == 'none':
            aspp_ini, decoder_ini, coarse_ini, gcn_ini = False, False, False, False

        if backbone == 'res2net':
            self.backbone = res2net.Res2Net101(output_stride=os, BatchNorm=BatchNorm, pretrained=pretrained)
        if backbone == 'resnet':
            self.backbone = resnet.ResNet101(output_stride=os, BatchNorm=BatchNorm, pretrained=pretrained)

        self.aspp = ASPP(backbone, os, BatchNorm, initial=aspp_ini)
        if sp_conv:
            convert_to_separable_conv(self.aspp, aspp_ini)

        if self.coarse_seg:
            self.coarse_logits = CoarseSeg(num_classes, BatchNorm, initial=coarse_ini)
            if sp_conv:
                convert_to_separable_conv(self.coarse_logits, coarse_ini)

        self.decoder = Decoder(num_classes, backbone, BatchNorm, initial=decoder_ini)
        if sp_conv:
            convert_to_separable_conv(self.decoder, decoder_ini)

        if self.use_gcn:
            self.self_att_gcn = NonLocalGCN(in_channels=256, BatchNorm=BatchNorm, inter_channels=256, gcn=self.gcn,
                                            k_ratio=k_ratio, initial=gcn_ini)

        self.criterion = getattr(criterion, 'bce_gdl')

    def forward(self, input, gt=None):

        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)

        if self.coarse_seg:
            coarse_logits = self.coarse_logits(x)  # b, c, h, w
            b, h, w = coarse_logits.shape[0], coarse_logits.shape[2], coarse_logits.shape[3]

        if self.use_gcn and self.coarse_seg:
            with torch.no_grad():
                K = int(self.np_ratio * h * w)
                # print('uncertainty nodes num:', K)
                hp_map = torch.zeros(b * h * w, 1, dtype=torch.long, device=coarse_logits.device)
                uncertainty_score = calculate_uncertainty(coarse_logits).view(b, 1, -1)[:, 0, :]  # b, 1, h, w -> b, hw
                idx = torch.topk(uncertainty_score, k=K, dim=1)[1]  # b, k
                shift = (h * w) * torch.arange(b, dtype=torch.long, device=coarse_logits.device)
                idx += shift[:, None]
                hp_map[idx.view(-1), :] = 1
                hp_map = hp_map.view(b, -1)  # b, n

            x = self.self_att_gcn(x, hp_map)

        x = self.decoder(x, low_level_feat)
        del low_level_feat

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        if self.coarse_seg:
            coarse_logits = F.interpolate(coarse_logits, size=input.size()[2:], mode='bilinear', align_corners=True)
            loss = self.criterion(x, gt) + self.criterion(coarse_logits, gt)
        else:
            loss = self.criterion(x, gt)
        return loss, x