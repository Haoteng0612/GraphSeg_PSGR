import sys
sys.path.append('../')
from os import listdir
from os.path import isfile, join


def write(data, fname, data_dir=None):
    fname = join(data_dir, fname)
    with open(fname, 'w') as f:
        f.write('\n'.join(data))


input_dir = '/media/userdisk0/hzjia/Data/covid100/slices/masks1'
out_dir = '/media/userdisk0/hzjia/Data/covid100/slices'

img_list = []
for img in listdir(input_dir):
    if isfile(join(input_dir, img)) and 'npy' in img:
        img_list.append(img)

write(img_list, 'all_list_new.txt', out_dir)

valid_study = [[92, 35, 34, 17, 38, 98, 16, 23, 90, 20, 0, 12, 57, 43, 6, 22, 46, 82, 13, 80],
               [8, 60, 41, 53, 65, 69, 96, 55, 97, 21, 66, 64, 68, 93, 62, 25, 49, 19, 39, 79],
               [26, 27, 67, 63, 71, 15, 95, 72, 3, 85, 14, 4, 77, 88, 56, 18, 59, 11, 2, 76],
               [94, 31, 47, 33, 24, 61, 29, 54, 81, 36, 28, 1, 9, 73, 99, 75, 58, 51, 44, 70],
               [40, 50, 74, 83, 10, 30, 45, 32, 84, 87, 37, 89, 78, 52, 91, 7, 5, 86, 42, 48]]

for i in range(len(valid_study)):
    valid_list = []
    for val_id in valid_study[i]:
        current_list1 = [k for k in img_list if str(val_id).zfill(2) in k]
        valid_list += current_list1
    valid_list = list(set(valid_list))
    train_list = list(set(img_list).difference(set(valid_list)))
    write(train_list, 'train_new{}.txt'.format(i), data_dir=out_dir)
    write(valid_list, 'valid_new{}.txt'.format(i), data_dir=out_dir)
