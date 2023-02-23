import torch
from torch.utils.data import Dataset
import numpy as np
from os.path import join


class TrainMosDataset(Dataset):
    def __init__(self, train_list, datadir=None, args=None, transforms=None):
        img_dir = join(datadir, 'imgs')
        label_dir = join(datadir, 'masks')
        names = []
        with open(join(datadir, train_list + '.txt')) as f:
            for line in f:
                line = line.strip()
                name = line
                names.append(name)
        # print('file list:' + str(names))
        self.names = names
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.args = args
        self.transforms = transforms

    def __getitem__(self, index):
        y = np.array(np.load(join(self.label_dir, self.names[index])), dtype='uint8', order='C')
        x = np.array(np.load(join(self.img_dir, self.names[index])), dtype='float32', order='C')
        # print(x.shape, y.shape)#(512, 512) (512, 512)
        x, y = self.transforms([x, y])
        x = x[..., None]        # (513, 513, 1)

        x = np.ascontiguousarray(x.repeat(3, 2).transpose(2, 0, 1))  # (3, 513, 513)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # (3, 513, 513) (513, 513)

        return x, y

    def __len__(self):
        return len(self.names)


class TrainCovid100Dataset_1(Dataset):
    def __init__(self, train_list, datadir=None, args=None, transforms=None):
        img_dir = join(datadir, 'imgs')
        label_dir = join(datadir, 'masks1')
        names = []
        with open(join(datadir, train_list + '.txt')) as f:
            for line in f:
                line = line.strip()
                name = line
                names.append(name)
        # print('file list:' + str(names))
        self.names = names
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.args = args
        self.transforms = transforms

    def __getitem__(self, index):
        y = np.array(np.load(join(self.label_dir, self.names[index])), dtype='uint8', order='C')
        x = np.array(np.load(join(self.img_dir, self.names[index])), dtype='float32', order='C')
        # print(x.shape, y.shape)#(512, 512) (512, 512)
        x, y = self.transforms([x, y])
        x = x[..., None]        # (513, 513, 1)

        x = np.ascontiguousarray(x.repeat(3, 2).transpose(2, 0, 1))  # (3, 513, 513)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # (3, 513, 513) (513, 513)

        return x, y

    def __len__(self):
        return len(self.names)

class TrainCovid100Dataset_3(Dataset):
    def __init__(self, train_list, datadir=None, args=None, transforms=None):
        img_dir = join(datadir, 'imgs')
        label_dir = join(datadir, 'masks3')
        names = []
        with open(join(datadir, train_list + '.txt')) as f:
            for line in f:
                line = line.strip()
                name = line
                names.append(name)
        # print('file list:' + str(names))
        self.names = names
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.args = args
        self.transforms = transforms

    def __getitem__(self, index):
        y = np.array(np.load(join(self.label_dir, self.names[index])), dtype='uint8', order='C')
        x = np.array(np.load(join(self.img_dir, self.names[index])), dtype='float32', order='C')
        # print(x.shape, y.shape)#(512, 512) (512, 512)
        x, y = self.transforms([x, y])
        x = x[..., None]        # (513, 513, 1)

        x = np.ascontiguousarray(x.repeat(3, 2).transpose(2, 0, 1))  # (3, 513, 513)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # (3, 513, 513) (513, 513)

        return x, y

    def __len__(self):
        return len(self.names)


class TrainCovid20Dataset(Dataset):
    def __init__(self, train_list, datadir=None, args=None, transforms=None):
        img_dir = join(datadir, 'imgs')
        label_dir = join(datadir, 'masks')
        names = []
        with open(join(datadir, train_list + '.txt')) as f:
            for line in f:
                line = line.strip()
                name = line
                names.append(name)
        # print('file list:' + str(names))
        self.names = names
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.args = args
        self.transforms = transforms

    def __getitem__(self, index):
        y = np.array(np.load(join(self.label_dir, self.names[index])), dtype='uint8', order='C')
        x = np.array(np.load(join(self.img_dir, self.names[index])), dtype='float32', order='C')
        # print('shape1:', x.shape, y.shape)
        # print(x.shape, y.shape)#(512, 512) (512, 512)
        x, y = self.transforms([x, y])
        x = x[..., None]        # (513, 513, 1)

        x = np.ascontiguousarray(x.repeat(3, 2).transpose(2, 0, 1))  # (3, 513, 513)
        y = np.ascontiguousarray(y)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        # print('shape2:', x.shape, y.shape)
        # (3, 513, 513) (513, 513)

        return x, y

    def __len__(self):
        return len(self.names)
