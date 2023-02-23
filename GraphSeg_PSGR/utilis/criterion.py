import torch
import torch.nn.functional as F


def bce_gdl(output, target, weight=1.0):
    """
    output: N,C,H,W
    target: N,H,W
    """
    if target is None:
        return 0
    else:
        if target.dim() == 3:
            target = expand_target(target, n_class=output.size()[1], mode='sigmoid')  # target: N,C,H,W
        loss = weight * F.binary_cross_entropy_with_logits(flatten(output), flatten(target))
        output = torch.sigmoid(output)
        loss += GeneralizedDiceLoss(output, target, weight_type='none', mode='sigmoid')
        return loss


def GeneralizedDiceLoss(output, target, eps=1e-5, weight_type='none', mode='sigmoid'):  # Generalized dice loss
    """
    Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentation'
    """
    if mode == 'softmax':
        output = output[:, 1:, :, :]
        target = target[:, 1:, :, :]
    axis_order = tuple(range(2, len(output.size())))
    target_sum = target.sum(axis_order)
    if weight_type == 'none':
        class_weights = 1.
    elif weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    # print(class_weights)
    intersect = (output * target).sum(axis_order, keepdim=False)
    intersect_sum = (intersect * class_weights).sum() + eps
    denominator = (output + target).sum(axis_order, keepdim=False)
    denominator_sum = (denominator * class_weights).sum() + eps

    dc = 2. * intersect_sum / denominator_sum
    return 1 - dc.mean()


def expand_target(x, n_class, mode='sigmoid'):
    """
        Converts NxHxW label image to NxCxHxW, where each label is stored in a separate channel
        :param input: 3D input image (NxHxW)
        :param C: number of channels/labels
        :return: 4D output image (NxCxHxW)
        """

    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if n_class == 1 :
        if mode.lower() == 'softmax':
            xx[:, 0, :, :] = (x == 0)
            xx[:, 1, :, :] = (x == 1)
        if mode.lower() == 'sigmoid':
            xx[:, 0, :, :] = (x == 1)
    else:
        if mode.lower() == 'softmax':
            xx[:, 0, :, :] = (x == 0)
            xx[:, 1, :, :] = (x == 1)
            xx[:, 2, :, :] = (x == 2)
            xx[:, 3, :, :] = (x == 3)
        if mode.lower() == 'sigmoid':
            xx[:, 0, :, :] = (x == 1)
            xx[:, 1, :, :] = (x == 2)
            xx[:, 2, :, :] = (x == 3)
    return xx.to(x.device)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is last.
    The shapes are transformed as follows:
       (N, C, H, W) -> (N * H * W, C)
    """
    C = tensor.size(1)

    # new axis order
    axis_order = (0,) + tuple(range(2, len(tensor.size()))) + (1,)
    transposed = tensor.permute(axis_order).contiguous() # Transpose: (N, C, H, W) -> (N, H, W, C)
    transposed = transposed.reshape(-1, C).contiguous()  # Flatten: (N, H, W, C) -> (N * H * W, C)
    return transposed
