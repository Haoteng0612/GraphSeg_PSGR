import sys
sys.path.append('../')
import numpy as np
import cv2
from utilis.utilis import check_mkdir
from os import listdir
from os.path import isfile, join


input_dir = '/media/userdisk0/hzjia/Data/ct_lesion_seg/mask'
output_dir = '/media/userdisk0/hzjia/Data/ct_lesion_seg/mask_visual'
check_mkdir(output_dir)

for patient in listdir(input_dir):
    patdir = join(input_dir, patient)
    out_patdir = join(output_dir, patient)
    check_mkdir(out_patdir)
    for img in listdir(patdir):
        if isfile(join(patdir, img)) and 'png' in img:
            name = img[:-4]
            array = np.array(cv2.imread(join(patdir, img)))
            array_new = np.zeros(array.shape)
            array_new[array == 0] = 0
            array_new[array == 1] = 85
            array_new[array == 2] = 170
            array_new[array == 3] = 255
            array_new.astype('uint8')
            cv2.imwrite(join(out_patdir, name + '.png'), array_new)
