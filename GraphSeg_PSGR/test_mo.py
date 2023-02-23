import argparse
from dataset.dataset import TrainDataset
from dataset import transforms as joint_transforms
from datetime import datetime
import os
from prefetch_generator import BackgroundGenerator
from ptflops import get_model_complexity_info
from models.network import GCNSegNet as seg_model
from models.unet2d import UNet as unet
from models.baselines import NestedUNet, ResUnet, AttU_Net
import torch.optim as optim
import numpy as np
import random
from tensorboardX import SummaryWriter
import torch
from eval_mo import validate, sigmoid_output_dice
import torch.backends.cudnn as cudnn
import torch.optim
from inference_mo import inference
from torch.utils.data import DataLoader
from utilis.utilis import check_mkdir, AverageMeter, replace_w_sync_bn
from models.sync_batchnorm import patch_replication_callback
import time

cudnn.benchmark = True


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


exp_name = 'GraphSeg'
previous_save_path = os.path.join('/media/userdisk0/hzjia/model_save', exp_name, '')
save_path = os.path.join('/media/userdisk0/hzjia/model_save', exp_name, datetime.now().strftime('%b%d_%H-%M-%S'))
runs_path = os.path.join('/media/userdisk0/hzjia/runs', exp_name, datetime.now().strftime('%b%d_%H-%M-%S'))
check_mkdir(save_path)
check_mkdir(runs_path)
writer = SummaryWriter(runs_path)


def main():
    parser = argparse.ArgumentParser(description='PyTorch GraphSeg Project')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--crop_size', type=int, default=513)
    parser.add_argument('--scales', type=int, default=[0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2])
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--nEpochs', type=int, default=500)
    parser.add_argument('--change_opt', action='store_true', default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--backbone', type=str, default='resnet', choices=('drn', 'resnet'))
    parser.add_argument('--embedded_module', type=str, default='spgr', choices=('spgr', 'non-local', 'glore', 'none'))
    parser.add_argument('--initial_type', type=str, default='none')
    parser.add_argument('--gcn', type=str, default='hogcn')
    parser.add_argument('--np_ratio', type=float, default=0.1)
    parser.add_argument('--k_ratio', type=float, default=0.5)
    parser.add_argument('--os', type=int, default=8)

    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'adam'))
    parser.add_argument('--resume', default='/media/userdisk0/hzjia/model_save/GraphSeg/Jan03_06-46-28/epoch_210_loss_0.659597_dice_0.698001')
    parser.add_argument('--valid_list', type=str, default='valid_4', choices=('valid_0', 'valid_1', 'valid_2',
                                                                              'valid_3', 'valid_4'))
    parser.add_argument('--datadir', type=str, default='/media/userdisk0/hzjia/Data/MosMedData/slices')

    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(args.gpu)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Network initialization
    model = seg_model(num_classes=1, os=args.os, embedded_module=args.embedded_module, gcn=args.gcn,
                      coarse_seg=True, pretrained=True, np_ratio=args.np_ratio, k_ratio=args.k_ratio,
                      backbone=args.backbone, initial=args.initial_type)
    # print("building segmentation model........", "backbone, os and initial type:", args.backbone,
    #       args.os, args.initial_type)
    #
    # model = unet(n_channels=3, n_classes=1, embedded_module=args.embedded_module, gcn=args.gcn, np_ratio=args.np_ratio,
    #              k_ratio=args.k_ratio, os=args.os)

    pretrained_weight = torch.load(os.path.join(previous_save_path, str(args.resume + '.pth')))
    model.load_state_dict(pretrained_weight['model_dict'])

    if args.cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        patch_replication_callback(model)

    # Loading dataset
    print('======> Loading valid datasets')
    valid_transforms = joint_transforms.Compose([
        joint_transforms.Pad(args.crop_size),
        joint_transforms.NumpyType((np.float32, np.uint8)),
    ])
    valid_set = TrainDataset(args.valid_list, datadir=args.datadir, args=args, transforms=valid_transforms)
    valid_loader = DataLoaderX(dataset=valid_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    # Network training
    with torch.no_grad():
        valid(args, model, valid_loader, valid_set.names)

def valid(args, model, validloader, names):
    dice_list = inference(validloader, model, args, names, scoring=True, use_TTA=False)
    # print(dice_list)
    # print('mean, std:', np.mean(dice_list), np.std(dice_list, ddof=0))

if __name__ == '__main__':
    main()
