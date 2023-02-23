import argparse
from dataset.dataset import TrainMosDataset
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
from torch.utils.data import DataLoader
from utilis.utilis import check_mkdir, AverageMeter, replace_w_sync_bn
from models.sync_batchnorm import patch_replication_callback
from PIL import Image
import time

cudnn.benchmark = True


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


exp_name = 'GraphSeg'
previous_save_path = os.path.join('', exp_name, '')
save_path = os.path.join('', exp_name, datetime.now().strftime('%b%d_%H-%M-%S'))
runs_path = os.path.join('', exp_name, datetime.now().strftime('%b%d_%H-%M-%S'))
check_mkdir(save_path)
check_mkdir(runs_path)
writer = SummaryWriter(runs_path)


def main():
    parser = argparse.ArgumentParser(description='PyTorch GraphSeg Project')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu', type=str, default='0,1')
    parser.add_argument('--crop_size', type=int, default=513)
    parser.add_argument('--scales', type=int, default=[0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2])
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--change_opt', action='store_true', default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--backbone', type=str, default='resnet', choices=('drn', 'resnet'))
    parser.add_argument('--embedded_module', type=str, default='spgr', choices=('spgr', 'non-local', 'glore', 'none'))
    parser.add_argument('--initial_type', type=str, default='none')
    parser.add_argument('--gcn', type=str, default='hogcn')
    parser.add_argument('--np_ratio', type=float, default=0.005)
    parser.add_argument('--k_ratio', type=float, default=0.5)
    parser.add_argument('--os', type=int, default=8)

    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'adam'))
    parser.add_argument('--resume', default='')
    parser.add_argument('--train_list', type=str, default='train_new0',
                        choices=('train_new0', 'train_new1', 'train_new2',
                                 'train_new3', 'train_new4'))
    parser.add_argument('--valid_list', type=str, default='valid_new0',
                        choices=('valid_new0', 'valid_new1', 'valid_new2',
                                 'valid_new3', 'valid_new4'))
    parser.add_argument('--datadir', type=str, default='')

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
    # model = seg_model(num_classes=1, os=args.os, embedded_module=args.embedded_module, gcn=args.gcn,
    #                   coarse_seg=True, pretrained=True, np_ratio=args.np_ratio, k_ratio=args.k_ratio,
    #                   backbone=args.backbone, initial=args.initial_type)
    # print("building segmentation model........", "backbone, os and initial type:", args.backbone,
    #       args.os, args.initial_type)

    model = unet(n_channels=3, n_classes=1, embedded_module=args.embedded_module, gcn=args.gcn, np_ratio=args.np_ratio,
                 k_ratio=args.k_ratio, os=args.os)
    print('using unet......')

    # model = AttU_Net(n_channels=3, n_classes=1)
    # print('building att-unet.........')

    # model = ResUnet(n_channels=3, n_classes=1)
    # print('building res-unet.........')

    # model = NestedUNet(n_channels=3, n_classes=1)
    # print('building unet++.........')

    if args.embedded_module == 'non-local':
        print("using traditional non-local self-attention......")
    if args.embedded_module == 'spgr':
        print("using the proposed spgr module.........")
        print("np_ratio:", args.np_ratio, "k_ratio:", args.k_ratio, "gcn_type:", args.gcn)
    if args.embedded_module == 'glore':
        print('using global reasoning module......')

    # model_info_file = open(os.path.join(os.getcwd(), str(args.crop_size) + '_model_info.txt'), 'w+')
    # flops, params = get_model_complexity_info(model, (3, 513, 513), as_strings=True, print_per_layer_stat=True,
    #                                           ost=model_info_file)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    if len(args.resume) == 0:
        curr_epoch = 1
        args.best_record = {'epoch': 0, 'valid_loss': 1, 'dice_value': 0}
    else:
        print('training resumes from' + args.resume)
        model.load_state_dict(torch.load(os.path.join(previous_save_path, str(args.resume + '.pth'))))
        split_snapshot = args.resume.split('_')
        curr_epoch = int(split_snapshot[1]) + 1

    # Optimization initialization
    if args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                              momentum=0.99, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.learning_rate, eps=args.eps, weight_decay=args.weight_decay)

    # if len(args.gpu.replace(',', '')) > 1 and args.batch_size // len(args.gpu.replace(',', '')) <= 4:
    #     model.apply(replace_w_sync_bn)
    #     use_sync_bn = True
    # else:
    #     use_sync_bn = False

    if args.cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        patch_replication_callback(model)

    # Transforms configuration
    train_transforms = joint_transforms.Compose([
        joint_transforms.Scale(args.scales),
        joint_transforms.Pad(args.crop_size),
        joint_transforms.RandomCrop(args.crop_size),
        joint_transforms.RandomRotion(15),
        joint_transforms.RandomIntensityChange((0.1, 0.1)),
        joint_transforms.RandomFlip(),
        joint_transforms.NumpyType((np.float32, np.uint8)),
    ])

    # Loading dataset
    print('======> Loading train datasets on:', args.train_list)
    train_set = TrainMosDataset(args.train_list, datadir=args.datadir, args=args, transforms=train_transforms)
    train_loader = DataLoaderX(dataset=train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                               pin_memory=False, shuffle=True)
    if args.valid_list is not None:
        print('======> Loading valid datasets')
        valid_transforms = joint_transforms.Compose([
            joint_transforms.Pad(args.crop_size),
            joint_transforms.NumpyType((np.float32, np.uint8)),
        ])
        valid_set = TrainMosDataset(args.valid_list, datadir=args.datadir, args=args, transforms=valid_transforms)
        valid_loader = DataLoaderX(dataset=valid_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    # Network training
    torch.set_grad_enabled(True)
    for epoch in range(curr_epoch, args.nEpochs + 1):
        adjust_learning_rate(optimizer, epoch, max_epoch=args.nEpochs, init_lr=args.learning_rate, warmup_epoch=5)
        train(args, epoch, model, train_loader, optimizer)
        if args.valid_list is not None and epoch % 10 == 0 and epoch != 0:
            with torch.no_grad():
                valid(args, epoch, model, valid_loader, optimizer)
                torch.cuda.empty_cache()


def train(args, epoch, model, trainloader, optimizer):
    model.train()
    nProcessed = 0
    nTrain = len(trainloader.dataset)
    ave_loss = AverageMeter()
    ave_dice = AverageMeter()
    for batch_idx, (data, target) in enumerate(trainloader):
        nProcessed += len(data)
        partialEpoch = epoch + batch_idx / len(trainloader) - 1

        if args.cuda:
            data, target = data.cuda(), target.cuda()
            target = target.long()

        loss, output = model(data, target)

        loss = loss.mean()
        optimizer.zero_grad()
        log_loss = loss.cpu().detach_()

        # measure dice and record loss
        output = torch.sigmoid(output)
        output_cpu = output.cpu().detach().numpy()
        output_cpu[output_cpu >= 0.5] = 1
        output_cpu[output_cpu < 0.5] = 0

        target = target.cpu().detach().numpy()
        dice_value = sigmoid_output_dice(output_cpu, target)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # update average loss
        ave_loss.update(log_loss.item())
        ave_dice.update(dice_value)
        # print the temporal result 1: Ground-glass opacity (GGO) and Consolidation (CO)
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}\t Dice: {:.6f}'.
              format(partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainloader), ave_loss.value(),
                     ave_dice.value()))

    # TensorboardX visualization
    writer.add_scalar('Train Loss', ave_loss.average(), epoch)
    writer.add_scalar('Train  Dice', ave_dice.average(), epoch)
    writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)


def valid(args, epoch, model, validloader, optimizer):
    score_avg = validate(epoch, validloader, model, args, writer, scoring=True, use_TTA=False, snapshot=True)

    dice_avg, loss_avg = score_avg[0], score_avg[1]

    # TnesorboardX visualization
    writer.add_scalar('Val Dice', score_avg[0], epoch)
    writer.add_scalar('Val Loss', score_avg[1], epoch)

    # save model in 50 epochs and the best model
    if dice_avg > args.best_record['dice_value']:
        args.best_record['epoch'] = epoch
        args.best_record['valid_loss'] = loss_avg
        args.best_record['dice_value'] = dice_avg
        resume_name = 'epoch_%d_loss_%.6f_dice_%.6f' % (epoch, loss_avg, dice_avg)
        torch.save({
            'epoch': epoch,
            'model_dict': model.module.state_dict(),
            'optim_dict': optimizer.state_dict(),
        }, os.path.join(save_path, resume_name + '.pth'))

    elif epoch % 50 == 0:
        resume_name = 'epoch_%d_loss_%.6f_dice_%.6f' % (epoch, loss_avg, dice_avg)
        torch.save({
            'epoch': epoch,
            'model_dict': model.module.state_dict(),
            'optim_dict': optimizer.state_dict(),
        }, os.path.join(save_path, resume_name + '.pth'))


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, warmup_epoch, power=0.9):
    if epoch < warmup_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = round(init_lr * min(1.0, epoch / warmup_epoch), 8)
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = round(init_lr * np.power(1 - (epoch - warmup_epoch) / max_epoch, power), 8)


if __name__ == '__main__':
    main()
