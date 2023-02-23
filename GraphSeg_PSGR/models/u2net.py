import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm import SynchronizedBatchNorm2d
from utilis import criterion
from utilis.utilis import calculate_uncertainty
from models.modules.non_local import NonLocal
from models.modules.self_attention_gcn import SPGR_Unit


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


class REBNCONV(nn.Module):
    def __init__(self,in_ch=3, out_ch=3, dirate=1, BatchNorm=nn.BatchNorm2d):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = BatchNorm(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear', align_corners=True)

    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, BatchNorm=nn.BatchNorm2d):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1, BatchNorm=BatchNorm)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2, BatchNorm=BatchNorm)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1, BatchNorm=BatchNorm)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, BatchNorm=nn.BatchNorm2d):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1, BatchNorm=BatchNorm)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2, BatchNorm=BatchNorm)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1, BatchNorm=BatchNorm)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, BatchNorm=nn.BatchNorm2d):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1, BatchNorm=BatchNorm)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2, BatchNorm=BatchNorm)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1, BatchNorm=BatchNorm)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, BatchNorm=nn.BatchNorm2d):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1, BatchNorm=BatchNorm)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2, BatchNorm=BatchNorm)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1, BatchNorm=BatchNorm)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, BatchNorm=nn.BatchNorm2d):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1, BatchNorm=BatchNorm)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1, BatchNorm=BatchNorm)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2, BatchNorm=BatchNorm)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4, BatchNorm=BatchNorm)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8, BatchNorm=BatchNorm)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4, BatchNorm=BatchNorm)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2, BatchNorm=BatchNorm)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1, BatchNorm=BatchNorm)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


##### U^2-Net ####
class U2NET(nn.Module):

    def __init__(self, n_channels=3, n_classes=1, sync_bn=True, embedded_module='spgr', gcn=False, np_ratio=0.01,
                 k_ratio=0.5, coarse_seg=True):
        super(U2NET,self).__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.embedded_module = embedded_module
        self.gcn = gcn
        self.np_ratio = np_ratio
        self.k_ratio = k_ratio
        self.coarse_seg = coarse_seg
        self.n_classes = n_classes

        if self.embedded_module == 'spgr':
            self.self_att_gcn = SPGR_Unit(in_channels=512, BatchNorm=BatchNorm, inter_channels=256,
                                          gcn=self.gcn, k_ratio=k_ratio, initial=False)
        elif self.embedded_module == 'non-local':
            self.non_local = NonLocal(in_channels=512, BatchNorm=BatchNorm, inter_channels=256,
                                      initial=False)
        elif self.embedded_module == 'glore':
            self.global_reasoning = GloRe_Unit(num_in=512, num_mid=64, BatchNorm=BatchNorm,
                                               initial=False)

        if self.coarse_seg:
            self.coarse_logits = nn.Sequential(ConvBNReLU(512, 256, BatchNorm, 3, 1, 1, 1),
                                               nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

        self.stage1 = RSU7(n_channels,32,64, BatchNorm)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128, BatchNorm)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256, BatchNorm)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512, BatchNorm)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512, BatchNorm)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512, BatchNorm)

        # decoder
        self.stage5d = RSU4F(1024,256,512, BatchNorm)
        self.stage4d = RSU4(1024,128,256, BatchNorm)
        self.stage3d = RSU5(512,64,128, BatchNorm)
        self.stage2d = RSU6(256,32,64, BatchNorm)
        self.stage1d = RSU7(128,16,self.n_classes, BatchNorm)

        # self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        # self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        # self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        # self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        # self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        # self.side6 = nn.Conv2d(512,out_ch,3,padding=1)
        #
        # self.outconv = nn.Conv2d(6*out_ch,out_ch,1)
        self.criterion = getattr(criterion, 'bce_gdl')

    def forward(self, x, gt=None):
        #stage 1
        hx1 = self.stage1(x)
        x = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(x)
        x = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(x)
        x = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(x)
        x = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(x)
        x = self.pool56(hx5)

        #stage 6
        x = self.stage6(x)
        x = _upsample_like(x,hx5)

        #-------------------- decoder --------------------
        x = self.stage5d(torch.cat((x,hx5),1))

        if self.coarse_seg:
            coarse_logits = self.coarse_logits(x)
            b, h, w = coarse_logits.shape[0], coarse_logits.shape[2], coarse_logits.shape[3]

            if self.embedded_module == 'spgr':
                with torch.no_grad():
                    K = int(self.np_ratio * h * w)
                    # print('uncertainty nodes num:', K)
                    hp_map = torch.zeros(b * h * w, 1, dtype=torch.long, device=coarse_logits.device)
                    uncertainty_score = calculate_uncertainty(coarse_logits, self.n_classes).view(b, 1, -1)[:, 0, :]  # b, 1, h, w -> b, hw
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

        x = _upsample_like(x,hx4)

        x = self.stage4d(torch.cat((x,hx4),1))
        x = _upsample_like(x,hx3)

        x = self.stage3d(torch.cat((x,hx3),1))
        x = _upsample_like(x,hx2)

        x = self.stage2d(torch.cat((x,hx2),1))
        x = _upsample_like(x,hx1)

        x = self.stage1d(torch.cat((x,hx1),1))

        # #side output
        # d1 = self.side1(hx1d)
        #
        # d2 = self.side2(hx2d)
        # d2 = _upsample_like(d2,d1)
        #
        # d3 = self.side3(hx3d)
        # d3 = _upsample_like(d3,d1)
        #
        # d4 = self.side4(hx4d)
        # d4 = _upsample_like(d4,d1)
        #
        # d5 = self.side5(hx5d)
        # d5 = _upsample_like(d5,d1)
        #
        # d6 = self.side6(hx6)
        # d6 = _upsample_like(d6,d1)
        #
        # x = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        if self.coarse_seg:
            coarse_logits = F.interpolate(coarse_logits, size=x.size()[2:], mode='bilinear', align_corners=True)
            loss = self.criterion(x, gt) + self.criterion(coarse_logits, gt) * 0.1
        else:
            loss = self.criterion(x, gt)
        return loss, x


# ### U^2-Net small ###
# class U2NETP(nn.Module):
#
#     def __init__(self,in_ch=3,out_ch=1):
#         super(U2NETP,self).__init__()
#
#         self.stage1 = RSU7(in_ch,16,64)
#         self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.stage2 = RSU6(64,16,64)
#         self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.stage3 = RSU5(64,16,64)
#         self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.stage4 = RSU4(64,16,64)
#         self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.stage5 = RSU4F(64,16,64)
#         self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
#
#         self.stage6 = RSU4F(64,16,64)
#
#         # decoder
#         self.stage5d = RSU4F(128,16,64)
#         self.stage4d = RSU4(128,16,64)
#         self.stage3d = RSU5(128,16,64)
#         self.stage2d = RSU6(128,16,64)
#         self.stage1d = RSU7(128,16,64)
#
#         self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
#         self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
#         self.side3 = nn.Conv2d(64,out_ch,3,padding=1)
#         self.side4 = nn.Conv2d(64,out_ch,3,padding=1)
#         self.side5 = nn.Conv2d(64,out_ch,3,padding=1)
#         self.side6 = nn.Conv2d(64,out_ch,3,padding=1)
#
#         self.outconv = nn.Conv2d(6*out_ch,out_ch,1)
#
#     def forward(self,x):
#         #stage 1
#         hx1 = self.stage1(x)
#         x = self.pool12(hx1)
#
#         #stage 2
#         hx2 = self.stage2(x)
#         x = self.pool23(hx2)
#
#         #stage 3
#         hx3 = self.stage3(x)
#         x = self.pool34(hx3)
#
#         #stage 4
#         hx4 = self.stage4(x)
#         x = self.pool45(hx4)
#
#         #stage 5
#         hx5 = self.stage5(x)
#         x = self.pool56(hx5)
#
#         #stage 6
#         hx6 = self.stage6(x)
#         hx6up = _upsample_like(hx6,hx5)
#
#         #decoder
#         hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
#         hx5dup = _upsample_like(hx5d,hx4)
#
#         hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
#         hx4dup = _upsample_like(hx4d,hx3)
#
#         hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
#         hx3dup = _upsample_like(hx3d,hx2)
#
#         hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
#         hx2dup = _upsample_like(hx2d,hx1)
#
#         hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))
#
#
#         #side output
#         d1 = self.side1(hx1d)
#
#         d2 = self.side2(hx2d)
#         d2 = _upsample_like(d2,d1)
#
#         d3 = self.side3(hx3d)
#         d3 = _upsample_like(d3,d1)
#
#         d4 = self.side4(hx4d)
#         d4 = _upsample_like(d4,d1)
#
#         d5 = self.side5(hx5d)
#         d5 = _upsample_like(d5,d1)
#
#         d6 = self.side6(hx6)
#         d6 = _upsample_like(d6,d1)
#
#         d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))
#
#         return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)