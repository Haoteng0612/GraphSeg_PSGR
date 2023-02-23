import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
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


def inference(
        valid_loader,
        model,
        args,
        names,
        scoring=True,  # If true, print the dice score.
        use_TTA=False,  # Test time augmentation, False as default!
):
    model.eval()
    dice_list = []

    for i, (data, target) in enumerate(valid_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
            target = target.long()

        loss, output = model(data, target)
        output = output.detach().cpu()
        target = target[0, ...].detach().cpu().numpy() if scoring else None

        # coarse_output = torch.sigmoid(coarse_output)
        # coarse_output = coarse_output.cpu().detach().numpy()[0, 0, ...]
        # coarse_output[coarse_output >= 0.5] = 1
        # coarse_output[coarse_output < 0.5] = 0
        #
        # gold_map = np.zeros(target.shape).astype(np.uint8)
        # gold_map[coarse_output != target] = 255
        # hp_map = hp_map.cpu().numpy().astype(np.uint8)[0, ...]
        # hp_map[hp_map > 0] = 255
        #
        # Image.fromarray(gold_map).save('path/' + names[i] + '_gold_map.png')
        # Image.fromarray(hp_map).save('path/' + names[i] + '_hp_map.png')

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

        if scoring:
            dice_value = sigmoid_output_dice(output, target)
            dice_list.append(dice_value)

    # computational_runtime(runtimes)
    return dice_list


def computational_runtime(runtimes):
    # remove the maximal value and minimal value
    runtimes = np.array(runtimes)
    maxvalue = np.max(runtimes)
    minvalue = np.min(runtimes)
    nums = runtimes.shape[0] - 2
    meanTime = (np.sum(runtimes) - maxvalue - minvalue) / nums
    fps = 1 / meanTime
    print('mean runtime:', meanTime, 'fps:', fps)
