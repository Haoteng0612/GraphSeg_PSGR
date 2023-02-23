import sys
sys.path.append('../')
import numpy as np
from os import listdir
import os
from os.path import isfile, join
from utilis.utilis import check_mkdir
import nibabel as nib


img_dir = '/media/userdisk0/hzjia/Data/covid20'
mask_dir = '/media/userdisk0/hzjia/Data/covid20/Infection_Mask'
out_img_dir = '/media/userdisk0/hzjia/Data/covid20/slices/imgs'
out_mask_dir = '/media/userdisk0/hzjia/Data/covid20/slices/masks'

check_mkdir(out_img_dir)
check_mkdir(out_mask_dir)

filename_list = [f for f in listdir(img_dir) if isfile(join(img_dir, f)) and 'nii.gz' in f]

def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])
    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data

slice_num = 0
for patient in filename_list:
    img = np.array(nib_load(join(img_dir, patient)), dtype='float32', order='C')
    mask = np.array(nib_load(join(mask_dir, patient)), dtype='uint8', order='C')

    n = 0
    for s in range(mask.shape[-1]):
        if np.sum(mask[:, :, s]) > 0:
            # print('foreground num:', np.sum(mask[:, :, s]))

            # pre-processing
            img_array = img[:, :, s]
            mean_v = img[:, :, s].mean()
            std_v = img[:, :, s].std()

            img_array -= mean_v
            img_array /= std_v

            np.save(join(out_img_dir, patient.replace('.nii.gz', '_' + str(s).zfill(4) + '.npy')), img_array)
            np.save(join(out_mask_dir, patient.replace('.nii.gz', '_' + str(s).zfill(4) + '.npy')), mask[:, :, s])
            n += 1
    slice_num += n
    print('finishing:', patient)

print('slice_num:', slice_num)

