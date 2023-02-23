import sys
sys.path.append('../')
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import StratifiedKFold


def write(data, fname, data_dir=None):
    fname = join(data_dir, fname)
    with open(fname, 'w') as f:
        f.write('\n'.join(data))


input_dir = '/media/userdisk0/hzjia/Data/MosMedData/slices/masks'
out_dir = '/media/userdisk0/hzjia/Data/MosMedData/slices'

img_list = []
for img in listdir(input_dir):
    if isfile(join(input_dir, img)) and 'npy' in img:
        img_list.append(img)

write(img_list, 'all_list.txt', out_dir)

X = img_list
Y = [0]*len(X)
X, Y = np.array(X), np.array(Y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for k, (train_index, valid_index) in enumerate(skf.split(Y, Y)):
    train_list = list(X[train_index])
    valid_list = list(X[valid_index])

    write(train_list, 'train_{}.txt'.format(k), data_dir=out_dir)
    write(valid_list, 'valid_{}.txt'.format(k), data_dir=out_dir)
