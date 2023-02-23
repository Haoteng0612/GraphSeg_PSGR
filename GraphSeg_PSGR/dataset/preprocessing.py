import sys

sys.path.append('../')
import numpy as np
from os import listdir
from os.path import isfile, join
import cv2
from utilis.utilis import check_mkdir


img_dir = '/media/userdisk0/hzjia/Data/ct_lesion_seg/image'
mask_dir = '/media/userdisk0/hzjia/Data/ct_lesion_seg/mask'
out_dir = '/media/userdisk0/hzjia/Data/ct_lesion_seg/image_norm'

for patient in listdir(mask_dir):
    imgdir = join(img_dir, patient)
    gtdir = join(mask_dir, patient)
    outdir = join(out_dir, patient)
    check_mkdir(outdir)
    for slice in listdir(gtdir):
        if isfile(join(gtdir, slice)) and 'png' in slice:
            name = slice[:-4]
            img_array = np.array(cv2.imread(join(imgdir, name + '.jpg')), dtype='float32', order='C')
            mean_v = img_array.mean()
            std_v = img_array.std()
            print('num:', img_array.shape)

            img_array -= mean_v
            img_array /= std_v
            np.save(join(outdir, name + '.npy'), img_array)
