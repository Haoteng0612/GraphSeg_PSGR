import torch
import torch.backends.cudnn as cudnn
import numpy as np
from utilis.utilis import AverageMeter

cudnn.benchmark = True


def dice_score(o, t, eps=1e-8):
    num = 2 * (o * t).sum() + eps  #
    den = o.sum() + t.sum() + eps  # eps
    # print('All_voxels: | numerator:{} | denominator:{} | pred_voxels:{} | GT_voxels:{}'.format(int(num), int(den),
    #                                                                                            o.sum(), int(t.sum())))
    return num / den


def sigmoid_output_dice(output, target):
    # 1: Ground-glass opacity (GGO) and Consolidation (CO)
    o = output[:, 0, :, :]
    t = (target == 1)
    ret = dice_score(o, t)

    return ret


def tta_output(model, data, target=None):
    _, output_logit = model(data, target)

    return output_logit


# 1:Ground-glass opacity (GGO) and Consolidation (CO)
keys = 'dice', 'loss'


def validate(
        epoch,
        valid_loader,
        model,
        args,
        writer,
        scoring=True,  # If true, print the dice score.
        use_TTA=False,  # Test time augmentation, False as default!
        snapshot=True,  # for visualization. Default false. It is recommended to generate the visualized figures.
):
    # H, W = args.crop_size, args.crop_size

    model.eval()
    ave_loss = AverageMeter()
    ave_dice = AverageMeter()

    for i, (data, target) in enumerate(valid_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            target = target.long()

        loss, output = model(data, target)
        output = output.detach().cpu()
        loss = loss.detach().cpu().numpy()
        target = target[0, ...].detach().cpu().numpy() if scoring else None

        # generate output
        if not use_TTA:
            output = torch.sigmoid(output)

        else:
            output = torch.sigmoid(output)
            output += torch.sigmoid(tta_output(model, data.flip(dims=(2,))).flip(dims=(2,)))
            output += torch.sigmoid(tta_output(model, data.flip(dims=(3,))).flip(dims=(3,)))
            output += torch.sigmoid(tta_output(model, data.flip(dims=(2, 3))).flip(dims=(2, 3)))
            output = output / 4.0  # mean

        output = output.detach().cpu().numpy()

        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        loss = loss.mean()

        if scoring:
            dice_value = sigmoid_output_dice(output, target)
            ave_loss.update(loss.item())
            ave_dice.update(dice_value)

            print('Subject {}/{}'.format(i + 1, len(valid_loader)))
            msg = 'epoch scores:'
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in
                              zip(keys, np.append(ave_dice.value(), ave_loss.value()))])
            print(msg)
            print('*******************************')

            if snapshot and epoch % 50 == 0:
                H, W = output.shape[-2:]
                # red: (1,0,0), tensorboardX requires the values lie in range of [0, 1]
                # 1 for foreground, and 0 for everything else.

                ins_width = 2  # pred and gt width = 2
                Snapshot_img = np.zeros(shape=(3, H, 2 * W + ins_width), dtype=np.uint8)
                # white boundary
                Snapshot_img[:, :, W:W + ins_width] = 1

                empty_fig = np.zeros(shape=(H, W), dtype=np.uint8)
                empty_fig[np.where(output[0, 0, ...] == 1)] = 1
                Snapshot_img[0, :, :W] = empty_fig
                empty_fig = np.zeros(shape=(H, W), dtype=np.uint8)
                empty_fig[np.where(target == 1)] = 1
                Snapshot_img[1, :, W + ins_width:2 * W + ins_width] = empty_fig

                Snapshot_img = np.array(Snapshot_img * 255, dtype='uint8')
                Snapshot_img = torch.from_numpy(Snapshot_img)

                # TnesorboardX visualization
                writer.add_image('epoch_%d_image_idx_%d_loss_%.6f_dice_%.6f' %
                                 (epoch, i, ave_loss.value(), ave_dice.value()), Snapshot_img)

    torch.cuda.empty_cache()
    if scoring:
        msg = 'Average scores:'
        msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in
                          zip(keys, np.append(ave_dice.average(), ave_loss.average()))])
        print(msg)

    # computational_runtime(runtimes)
    return [ave_dice.average(), ave_loss.average()]


def computational_runtime(runtimes):
    # remove the maximal value and minimal value
    runtimes = np.array(runtimes)
    maxvalue = np.max(runtimes)
    minvalue = np.min(runtimes)
    nums = runtimes.shape[0] - 2
    meanTime = (np.sum(runtimes) - maxvalue - minvalue) / nums
    fps = 1 / meanTime
    print('mean runtime:', meanTime, 'fps:', fps)
