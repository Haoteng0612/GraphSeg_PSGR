import torch.nn as nn
import torch
from models.sync_batchnorm import SynchronizedBatchNorm2d


class NonLocal(nn.Module):
    def __init__(self, in_channels, BatchNorm, inter_channels=None, sub_sample=False, bn_layer=True, initial=True):
        super(NonLocal, self).__init__()

        print('initial non local ...........')
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        if inter_channels is None:
            self.inter_channels = in_channels // 2
        else:
            self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.w = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                BatchNorm(self.in_channels)
            )

        else:
            self.w = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

        if initial:
            self._init_weight()

    def forward(self, x):

        b = x.size(0)

        g_x = self.g(x).view(b, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1).contiguous()  # b, hw, c

        theta_x = self.theta(x).view(b, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1).contiguous()  # b, hw, c
        phi_x = self.phi(x).view(b, self.inter_channels, -1)  # b, c, hw

        att_soft = torch.matmul(theta_x, phi_x)  # b, hw, hw
        del phi_x, theta_x

        att_soft = torch.softmax(att_soft, dim=-1)

        g_x = torch.matmul(att_soft, g_x)
        g_x = g_x.view(b, *x.size()[2:], -1)  # b, h, w, c
        g_x = g_x.permute(0, 3, 1, 2).contiguous()  # b, c, h, w
        g_x = self.w(g_x)

        x = x + g_x
        del g_x

        return x

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