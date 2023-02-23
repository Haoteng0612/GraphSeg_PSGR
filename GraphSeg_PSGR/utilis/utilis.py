import numpy as np
from models.sync_batchnorm import SynchronizedBatchNorm2d
import os
import torch.nn as nn
import torch


root = '/media/userdisk0/hzjia/PretrainedModels/'
drn54_pth = os.path.join(root, 'DRN', 'drn_d_54-0e0534ff.pth')
res50_pth = os.path.join(root, 'ResNet', 'resnet50-19c8e357.pth')
res101_pth = os.path.join(root, 'ResNet', 'resnet101-5d3b4d8f.pth')
res2net101_pth = os.path.join(root, 'ResNet', 'res2net101_26w_4s-02a759a1.pth')
res2net101_v1b_26w_4s_pth = os.path.join(root, 'Res2Net', 'res2net101_v1b_26w_4s-0812c246.pth')
res2net50_pth = os.path.join(root, 'ResNet', 'res2net50_26w_4s-06e79181.pth')
res152_pth = os.path.join(root, 'ResNet', 'resnet152-b121ed2d.pth')
inception_v3_pth = os.path.join(root, 'Inception', 'inception_v3_google-1a9a5a14.pth')
vgg19_bn_pth = os.path.join(root, 'VggNet', 'vgg19_bn-c79401a0.pth')
vgg16_pth = os.path.join(root, 'VggNet', 'vgg16-397923af.pth')
dense201_pth = os.path.join(root, 'DenseNet', 'densenet201-4c113574.pth')


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def replace_w_sync_bn(m):
    for var_name in dir(m):
        target_attr = getattr(m, var_name)
        if type(target_attr) == torch.nn.BatchNorm2d:
            num_features = target_attr.num_features
            eps = target_attr.eps
            momentum = target_attr.momentum
            affine = target_attr.affine

            # get parameters
            running_mean = target_attr.running_mean
            running_var = target_attr.running_var
            if affine:
                weight = target_attr.weight
                bias = target_attr.bias

            setattr(m, var_name,
                    SynchronizedBatchNorm2d(num_features, eps, momentum, affine))

            target_attr = getattr(m, var_name)
            # set parameters
            target_attr.running_mean = running_mean
            target_attr.running_var = running_var
            if affine:
                target_attr.weight = weight
                target_attr.bias = bias

    for var_name, children in m.named_children():
        replace_w_sync_bn(children)


def subdirs(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def dice_score(o, t, eps=1e-8):
    num = 2 * (o * t).sum() + eps  #
    den = o.sum() + t.sum() + eps  # eps
    # print('All_voxels: | numerator:{} | denominator:{} | pred_voxels:{} | GT_voxels:{}'.format(int(num), int(den),
    #                                                                                            o.sum(), int(t.sum())))
    return num / den


def sigmoid_output_dice(output, target):
    ret = []
    # 1: Lung field (LF)  2: Ground-glass opacity (GGO) 3: Consolidation (CO)
    # 1
    o = output[:, 0, :, :]
    t = (target == 1)
    ret.append(dice_score(o, t))
    # 2
    o = output[:, 1, :, :]
    t = (target == 2)
    ret.append(dice_score(o, t))
    # 3
    o = output[:, 2, :, :]
    t = (target == 3)
    ret.append(dice_score(o, t))
    return ret


def softmax_output_dice(output, target):
    ret = []
    # 1: Lung field (LF)  2: Ground-glass opacity (GGO) 3: Consolidation (CO)
    # 1
    o = (output == 1)
    t = (target == 1)
    ret.append(dice_score(o, t))
    # 2
    o = (output == 2)
    t = (target == 2)
    ret.append(dice_score(o, t))
    # 3
    o = (output == 3)
    t = (target == 3)
    ret.append(dice_score(o, t))

    return ret


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


class FullModel(nn.Module):
    """
          Distribute the loss on multi-gpu to reduce
          the memory cost in the main gpu.
          You can check the following discussion.
          https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
          """

    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, inputs, labels):
        outputs = self.model(inputs)
        loss = self.loss(outputs, labels)
        return loss, outputs


#
# # test num_workers and pin_memory
# train_set = TrainDataset(args.train_list, datadir=args.datadir, args=args)
#
# pin_memory = True
# print('pin_memory is', pin_memory)
# for num_workers in range(4, 32, 1):
#     training_dataloader = DataLoaderX(dataset=train_set, batch_size=args.batch_size,
#                                       num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)
#     start = time.time()
#     for epoch in range(1, 3):
#         for i, data in enumerate(training_dataloader, 0):
#             pass
#     end = time.time()
#     print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
# exit()


def sparse_graph(x, A, target, K_ratio=0.5):
    '''
    For computing the sparse graph adjacency and corresponding node features. N=HW. All tensors in batch
    :param x: original node features. B * N * C
    :param adj: original fully connected weighted adjacency matrix. B * N * N
    :param target: Hard point node index. B * N  vector.
    :param K: sparse ratio
    :return: x=x (B * N * C). Sparse Matrix adj_ (B * N * N)
    '''
    # A = torch.exp_(torch.neg(adj))  # A normalized
    num_batch, num_node = A.shape[0], A.shape[1]
    K = int(K_ratio * num_node)
    # create torch tensor to save sparse adj matrix
    A_sparse = torch.zeros([num_batch, num_node, num_node], dtype=torch.float, requires_grad=False).to(
        device=A.device)  # B * N * N

    # normalize A, set diag(A) = 0
    diag = torch.eye(num_node, num_node, requires_grad=False).to(device=A.device)  # b, n, n

    A = A - A * diag

    # compute D^(-1)
    D = torch.sum(A, dim=2)
    D = torch.pow(D, -1.0)
    D_ = torch.diag_embed(D, dim1=-2, dim2=-1)  # b, n, n

    # compute connection probability matrix L
    L = torch.bmm(D_, A)  # b, n, n    e.g. line 1 represents the prob. from node 1 to each of any other nodes

    # start to reconstruct the sparse matrix
    target_index = target == 1  # b, n
    target_index = target_index.unsqueeze(-1).repeat(1, 1, num_node).float()  # b, n, n

    score_support = torch.abs(L) * torch.norm(x, p=1, dim=-1).unsqueeze(1)  # b, n, n * b, 1, n -> b, n, n

    topk, indices = torch.topk(score_support, k=K, dim=2)  # b, n, k
    del score_support

    top_index = torch.zeros([num_batch, num_node, num_node], dtype=torch.float, requires_grad=False).to(
        device=A.device)  # B * N * N

    top_index = top_index.scatter(2, indices, topk)

    A_sparse[top_index != 0] = A[top_index != 0]
    del top_index

    A_sparse = A_sparse * target_index
    A_sparse = A_sparse + A_sparse.permute(0, 2, 1).contiguous()

    return A_sparse

    # num_batch, num_node = A.shape[0], A.shape[1]
    # diag = torch.eye(num_node, num_node, requires_grad=False).to(device=A.device)  # b, n, n
    # A = A - A * diag
    # # start to reconstruct the sparse matrix
    # target_index = target == 1  # b, n
    # target_index = target_index.unsqueeze(-1).repeat(1, 1, num_node).float()  # b, n, n
    # A = A * target_index
    # A = A + A.permute(0, 2, 1).contiguous()
    #
    # return A


def calculate_uncertainty(sem_seg_logits, n_class=3):
    """
    For each location of the prediction `sem_seg_logits` we estimate uncerainty as the
        difference between top first and top second predicted logits.
    Args:
        mask_logits (Tensor): A tensor of shape (N, C, ...), where N is the minibatch size and
            C is the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (N, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    if n_class > 1:
        top2_scores = torch.topk(sem_seg_logits, k=2, dim=1)[0]
        return (top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1)
    else:
        uncertainty_score = torch.sigmoid(sem_seg_logits)
        return 1 - torch.abs(uncertainty_score - 0.5)


def grid_2_COO(x, att_soft, hard_map, k_ratio):
    '''
    create the batch concatenated node feature x and COO edge_index
    :param x: node features g_x, dim is bhw * c, hw is the # of nodes and c is fea dim
    :param hard_map: hard nodes index, dim is b * hw
    :param adj: graph adjacency matrix, dim is b * hw * hw.
    :return: batch concat fea matrix (bhw * c) and batch concat COO edge_index (2 * # of connections)
    '''

    adj_sparse = sparse_graph(x, att_soft, hard_map, k_ratio)
    #### check is the batch size for x and adj is the same
    # adj_sparse = torch.exp_(torch.neg(adj_sparse))
    batch_x, batch_adj = x.shape[0], adj_sparse.shape[0]
    if batch_x != batch_adj:
        print("batch size is not matched !!")
        raise ValueError
    else:
        b = int(batch_x)  # b

    num_node = int(x.shape[1])  # hw

    #### process adj
    # set diag to 0
    diag = torch.eye(num_node, num_node).to(device=adj_sparse.device)  # b, hw, hw

    adj_sparse = adj_sparse - adj_sparse * diag
    del diag

    start_node = torch.arange(b, dtype=torch.long, device=adj_sparse.device) * num_node

    start_node = start_node.unsqueeze(-1).unsqueeze(-1) * torch.ones(adj_sparse.shape, dtype=torch.long,
                                                                     device=adj_sparse.device)  # b, n, n
    start_node = start_node[adj_sparse != 0]  # m
    start_node = start_node.repeat(2, 1)  # 2, m

    # obtain edge index
    edge_index = torch.nonzero(adj_sparse, as_tuple=False)  # m, 3
    edge_index = edge_index.permute(1, 0).contiguous()  # 3, m

    # obtain edge weight
    edge_weight = adj_sparse[adj_sparse != 0]  # m

    if edge_weight.shape[0] != edge_index.shape[1]:
        print("dim of edge_weight and dim of edge_index are not matched !")
        print("shape of edge_weight and edge_index are {}{}".format(edge_weight.shape, edge_index.shape))
        raise ValueError
    edge_weight = edge_weight.view(1, -1)  # 1, m

    # obtain edge index weight
    edge_index_weight = torch.cat((edge_index[1:], edge_weight))  # 3, m

    # concat the batch edge_index
    edge_index_weight[0:2, ...] += start_node

    par_indices = edge_index_weight[0:2, :]
    par_values = edge_index_weight[2, :]

    return par_indices.long(), par_values.float()

#
# # test num_workers and pin_memory
#     train_set = TrainDataset(args.train_list, datadir=args.datadir, args=args)
#
#     pin_memory = False
#     print('pin_memory is', pin_memory)
#     for num_workers in range(4, 32, 1):
#         training_dataloader = DataLoaderX(dataset=train_set, batch_size=args.batch_size,
#                                           num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)
#         start = time.time()
#         for epoch in range(0, 2):
#             for i, data in enumerate(training_dataloader, 0):
#                 pass
#         end = time.time()
#         print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
#
#     pin_memory = True
#     print('pin_memory is', pin_memory)
#     for num_workers in range(4, 32, 1):
#         training_dataloader = DataLoaderX(dataset=train_set, batch_size=args.batch_size,
#                                           num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)
#         start = time.time()
#         for epoch in range(0, 2):
#             for i, data in enumerate(training_dataloader, 0):
#                 pass
#         end = time.time()
#         print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
#     exit()
