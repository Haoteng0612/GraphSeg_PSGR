import sys
sys.path.append('../')
import numpy as np
from os import listdir
import os
from os.path import isfile, join
from utilis.utilis import check_mkdir
import nibabel as nib


img_dir = '/media/userdisk0/hzjia/Data/covid100'
out_img_dir = '/media/userdisk0/hzjia/Data/covid100/slices/imgs'
out_mask_dir1 = '/media/userdisk0/hzjia/Data/covid100/slices/masks1'
out_mask_dir2 = '/media/userdisk0/hzjia/Data/covid100/slices/masks3'

check_mkdir(out_img_dir)
check_mkdir(out_mask_dir1)
check_mkdir(out_mask_dir2)

def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])
    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data

img = np.array(nib_load(join(img_dir, 'tr_im.nii.gz')), dtype='float32', order='C')
mask = np.array(nib_load(join(img_dir, 'tr_mask.nii.gz')), dtype='uint8', order='C')

for s in range(mask.shape[-1]):
    print('foreground num:', np.sum(mask[:, :, s]))
    # pre-processing
    img_array = img[:, :, s]
    mean_v = img[:, :, s].mean()
    std_v = img[:, :, s].std()

    img_array -= mean_v
    img_array /= std_v

    mask3 = mask[:, :, s]
    mask3[mask3 > 0] = 1

    np.save(join(out_img_dir, 'tr_im_' + str(s).zfill(2) + '.npy'), img_array)
    np.save(join(out_mask_dir1, 'tr_im_' + str(s).zfill(2) + '.npy'), mask[:, :, s])
    np.save(join(out_mask_dir2, 'tr_im_' + str(s).zfill(2) + '.npy'), mask3)
    print('finishing:', s)


