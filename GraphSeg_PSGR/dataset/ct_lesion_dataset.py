import torch
from torch.utils.data import Dataset
import numpy as np
from os.path import join
import cv2
import torch.nn.functional as F
from scipy.ndimage import rotate
# from PIL import Image
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def scale(image, label=None, scales=None):
    if np.random.rand() < 0.5:
        ratio = np.random.choice(scales)
        image = F.interpolate(image, scale_factor=ratio, mode='bilinear', align_corners=True,
                              recompute_scale_factor=True)
        if label is not None:
            label = F.interpolate(label, scale_factor=ratio, mode='nearest',
                                  recompute_scale_factor=True)

    return image, label


def pad(image, label=None, crop_size=None):
    h, w = image.size()[-2:]
    pad_h = max(crop_size - h, 0)
    pad_w = max(crop_size - w, 0)
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0.)
        if label is not None:
            label = F.pad(label, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return image, label


def crop(image, label=None, crop_size=None):
    h, w = image.size()[-2:]
    s_h = np.random.randint(0, h - crop_size + 1)
    s_w = np.random.randint(0, w - crop_size + 1)
    e_h = s_h + crop_size
    e_w = s_w + crop_size
    image = image[:, :, s_h: e_h, s_w: e_w]
    label = label[:, :, s_h: e_h, s_w: e_w]
    return image, label


def rotation(image, label, angle_spectrum=15, axes=(0, 1)):
    image = rotate(image, angle_spectrum, axes=axes, reshape=False, order=0, mode='constant', cval=-1)
    label = rotate(label, angle_spectrum, axes=axes, reshape=False, order=0, mode='constant', cval=-1)

    return image, label


def flip(image, label=None):
    if np.random.rand() < 0.5:
        image = torch.flip(image, [2])
        if label is not None:
            label = torch.flip(label, [2])
    return image, label


class TrainDataset(Dataset):
    def __init__(self, train_list, datadir=None, args=None, transforms=None):
        img_dir = join(datadir, 'image_norm')
        label_dir = join(datadir, 'mask')
        names = []
        with open(join(datadir, train_list + '.txt')) as f:
            for line in f:
                line = line.strip()
                name = line[:-4]
                names.append(name)
        # print('file list:' + str(names))
        self.names = names
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.args = args
        self.transforms = transforms

    def __getitem__(self, index):
        # y = np.array(Image.open(join(self.label_dir, self.names[index] + '.png')), dtype='uint8', order='C')

        y = np.array(cv2.imread(join(self.label_dir, self.names[index] + '.png')), dtype='uint8', order='C')
        y = y[:, :, 0]
        x = np.array(np.load(join(self.img_dir, self.names[index] + '.npy')), dtype='float32', order='C')
        x = x[:, :, 0]
        # print(x.shape, y.shape)#(513, 513) (513, 513)

        x, y = self.transforms([x, y])
        x = x[..., None]        # (513, 513, 1)

        x = np.ascontiguousarray(x.repeat(3, 2).transpose(2, 0, 1))  # (3, 513, 513)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # (3, 513, 513) (513, 513)

        return x, y

    def __len__(self):
        return len(self.names)


# class ValDataset(Dataset):
#     def __init__(self, val_list, datadir=None, args=None, transforms=None):
#         img_dir = join(datadir, 'image_norm')
#         label_dir = join(datadir, 'mask')
#         names = []
#         with open(join(datadir, val_list + '.txt')) as f:
#             for line in f:
#                 line = line.strip()
#                 name = line[:-4]
#                 names.append(name)
#         # print('file list:' + str(names))
#         self.names = names
#         self.label_dir = label_dir
#         self.img_dir = img_dir
#         self.args = args
#         self.transforms = transforms
#
#     def __getitem__(self, index):
#         # y = np.array(Image.open(join(self.label_dir, self.names[index] + '.png')), dtype='uint8', order='C')
#         y = np.array(cv2.imread(join(self.label_dir, self.names[index] + '.png')), dtype='uint8', order='C')
#         y = y[:, :, 0]
#         x = np.array(np.load(join(self.img_dir, self.names[index] + '.npy')), dtype='float32', order='C')
#         x = x[:, :, 0]
#         # print(x.shape, y.shape)#(513, 513) (513, 513)
#
#         x, y = self.transforms([x, y])
#         x = x[..., None]  # (513, 513, 1)
#
#         x = np.ascontiguousarray(x.repeat(3, 2).transpose(2, 0, 1))  # (3, 513, 513)
#         y = np.ascontiguousarray(y)
#         x, y = torch.from_numpy(x), torch.from_numpy(y)
#         # print(x.shape, y.shape)  # (3, 513, 513) (513, 513)
#
#         return x, y
#
#     def __len__(self):
#         return len(self.names)
