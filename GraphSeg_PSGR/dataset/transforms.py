# import math
import random
import collections
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import zoom
import warnings

warnings.filterwarnings('ignore', '.*output shape of zoom.*')


class Base(object):
    def sample(self, *shape):
        return shape

    def tf(self, img, k=0):
        return img

    def __call__(self, img, reuse=False):  # class -> func()
        # image: nhwtc
        # shape: no first dim
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            shape = im.shape
            # print(shape) # (512, 512)
            self.sample(*shape)

        if isinstance(img, collections.Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)]  # img:k=0,label:k=1

        return self.tf(img)

    def __str__(self):
        return 'Identity()'


Identity = Base


# geometric transformations, need a buffers
class RandomRotion(Base):
    def __init__(self, angle_spectrum=15):
        assert isinstance(angle_spectrum, int)
        axes = (0, 1)
        self.angle_spectrum = angle_spectrum
        self.axes_buffer = axes

    def sample(self, *shape):
        if random.random() < 0.5:
            if random.random() < 0.75:
                self.angle_buffer = random.randint(1, 3) * 90
            else:
                self.angle_buffer = 0
        else:
            self.angle_buffer = np.random.randint(-self.angle_spectrum, self.angle_spectrum)
        return list(shape)

    def tf(self, img, k=0):
        """ Introduction: The rotation function supports the shape [H,W]
        :param img:
        :param k: if x, k=0; if label, k=1
        """
        img = rotate(img, self.angle_buffer, axes=self.axes_buffer, reshape=False, order=0,
                     mode='constant', cval=-1)

        return img

    def __str__(self):
        return 'RandomRotion(axes={},Angle:{}'.format(self.axes_buffer, self.angle_buffer)


class Scale(Base):
    def __init__(self, scale_factor=[0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2]):

        self.s_f = np.random.choice(scale_factor)
        self.scale_buffer = np.random.choice([True, False])

    def sample(self, *shape):
        # shape: (128， 128， 64)
        if self.scale_buffer:
            shape = [np.round(i * self.s_f) for i in shape]
            return shape
        else:
            return list(shape)

    def tf(self, img, k=0):
        """ Introduction: The scale function supports the shape [H,W,D]
        :param img: shape is [H,W,D]
        """
        if self.scale_buffer:
            img = zoom(img, (self.s_f, self.s_f), order=0, mode='nearest')

        return img

    def __str__(self):
        return 'Scale(scale_factor:{}'.format(self.s_f)


class RandomFlip(Base):
    # mirror flip across all x,y,z
    def __init__(self, ):
        self.axis = (0, 1)
        self.x_buffer = None
        self.y_buffer = None

    def sample(self, *shape):
        self.x_buffer = np.random.choice([True, False])
        self.y_buffer = np.random.choice([True, False])
        return list(shape)  # the shape is not changed

    def tf(self, img, k=0):  # img shape is (128, 128, 64)
        if self.x_buffer:
            img = np.flip(img, axis=self.axis[0])
        if self.y_buffer:
            img = np.flip(img, axis=self.axis[1])
        return img


class CenterCrop(Base):
    def __init__(self, patch_size):
        self.patch_size = patch_size
        self.buffer = None
        self.cropping = False

    def sample(self, *shape):
        if shape[0] > self.patch_size:
            self.cropping = True
            patch_size = [self.patch_size, self.patch_size]
            start = [(s - i) // 2 for i, s in zip(patch_size, shape)]
            self.buffer = [slice(s, s + k) for k, s in zip(patch_size, start)]
            shape = patch_size
            return shape
        return list(shape)

    def tf(self, img, k=0):
        if self.cropping:
            return img[tuple(self.buffer)]
        else:
            return img

    def __str__(self):
        return 'CenterCrop({})'.format(self.size)


class RandomCrop(Base):
    def __init__(self, patch_size):
        self.patch_size = patch_size
        self.buffer = []
        self.cropping = False

    def sample(self, *shape):
        self.cropping = shape[0] > self.patch_size or shape[1] > self.patch_size
        if self.cropping:
            patch_size = [self.patch_size, self.patch_size]
            start = [random.randint(0, s - i) for i, s in zip(patch_size, shape)]
            self.buffer = [slice(s, s + k) for k, s in zip(patch_size, start)]
            shape = patch_size
            # print(shape, self.buffer)
            return shape
        return list(shape)

    def tf(self, img, k=0):
        if self.cropping:
            # self.cropping = False
            img = img[tuple(self.buffer)]
            return img
        else:
            return img

    def __str__(self):
        return 'RandomCrop({})'.format(self.size)


# for data only
class RandomIntensityChange(Base):
    def __init__(self, factor):
        shift, scale = factor
        assert (shift > 0) and (scale > 0)
        self.shift = shift
        self.scale = scale

    def tf(self, img, k=0):
        if k == 1:
            return img

        shift_factor = np.random.uniform(-self.shift, self.shift)  # [-0.1,+0.1]
        scale_factor = np.random.uniform(1.0 - self.scale, 1.0 + self.scale)  # [0.9,1.1)
        return img * scale_factor + shift_factor

    def __str__(self):
        return 'random intensity shift per channels on the input image, including'


class Pad(Base):
    def __init__(self, patch_size):
        self.patch_size = patch_size
        self.padding = False
        self.pad = None
        self.px = None

    def sample(self, *shape):
        shape = list(shape)
        # print('pad shape:', shape, shape[0] < self.patch_size or shape[1] < self.patch_size)
        self.padding = shape[0] < self.patch_size or shape[1] < self.patch_size
        if self.padding:
            patch_size = [self.patch_size, self.patch_size]
            padx = patch_size[0] - shape[0] if patch_size[0] - shape[0] > 0 else 0
            pady = patch_size[1] - shape[1] if patch_size[1] - shape[1] > 0 else 0
            self.pad = (padx, pady)
            shape = shape[0] + padx, shape[1] + pady

        return shape

    def tf(self, img, k=0):
        # nhwdc, nhwd
        if self.padding:
            self.px = tuple(zip([0] * len(self.pad), self.pad))
            return np.pad(img, self.px, mode='constant')
        else:
            return img

    def __str__(self):
        return 'Pad(({}, {}, {}))'.format(*self.pad)


class GammaTransform(Base):
    def __init__(self, gamma_range):
        if np.random.random() < 0.5:
            self.gamma = np.random.uniform(gamma_range[0], 1)
        else:
            self.gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])

    def tf(self, img, k=0):
        if random.random() < 0.25:
            minm = img.min()
            rnge = img.max() - minm
            img = np.power(((img - minm) / float(rnge + 1e-8)), self.gamma) * rnge + minm
        return img

    def __str__(self):
        return 'Gamma()'


class GaussianNoise(Base):
    def __init__(self, sigma):
        self.sigma = np.random.uniform(sigma[0], sigma[1])

    def tf(self, img, k=0):
        if k > 0:
            return img
        else:
            if random.random() < 0.25:
                shape = img.shape
                img = img * np.exp(self.sigma * torch.randn(shape, dtype=torch.float32).numpy())
            return img

    def __str__(self):
        return 'Noise()'


class GaussianBlur(Base):
    def __init__(self, sigma):
        self.sigma = np.random.uniform(sigma[0], sigma[1])

    def tf(self, img, k=0):
        if k > 0:
            return img
        if random.random() < 0.25:
            img = ndimage.gaussian_filter(img, self.sigma)
        return img

    def __str__(self):
        return 'GaussianBlur()'


class ToNumpy(Base):
    def __init__(self, num=-1):
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        return img.numpy()

    def __str__(self):
        return 'ToNumpy()'


class ToTensor(Base):
    def __init__(self, num=-1):
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img

        return torch.from_numpy(img)

    def __str__(self):
        return 'ToTensor'


class TensorType(Base):
    def __init__(self, types, num=-1):
        self.types = types  # ('torch.float32', 'torch.int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.type(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'TensorType(({}))'.format(s)


class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types  # ('float32', 'int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'NumpyType(({}))'.format(s)


class Normalize(Base):
    def __init__(self, mean=0.0, std=1.0, num=-1):
        self.mean = mean
        self.std = std
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        img -= self.mean
        img /= self.std
        return img

    def __str__(self):
        return 'Normalize()'


class Compose(Base):
    def __init__(self, ops):
        if not isinstance(ops, collections.Sequence):
            ops = ops,
        self.ops = ops

    def sample(self, *shape):
        for op in self.ops:
            shape = op.sample(*shape)

    def tf(self, img, k=0):
        # is_tensor = isinstance(img, torch.Tensor)
        # if is_tensor:
        #    img = img.numpy()

        for op in self.ops:
            # print(op,img.shape,k)
            img = op.tf(img, k)  # do not use op(img) here

        # if is_tensor:
        #    img = np.ascontiguousarray(img)
        #    img = torch.from_numpy(img)

        return img

    def __str__(self):
        ops = ', '.join([str(op) for op in self.ops])
        return 'Compose([{}])'.format(ops)


# class Uniform(object):
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b
#
#     def sample(self):
#         return random.uniform(self.a, self.b)
#
#
# class Gaussian(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std
#
#     def sample(self):
#         return random.gauss(self.mean, self.std)
#
#
# class Constant(object):
#     def __init__(self, val):
#         self.val = val
#
#     def sample(self):
#         return self.val
