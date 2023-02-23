import sys
sys.path.append('../')
from os import listdir
from os.path import isfile, join


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

write(img_list, 'all_list_new.txt', out_dir)

valid_study = [[259, 283, 276, 261, 258, 269, 263, 281, 293, 301],
               [300, 267, 256, 262, 286, 275, 268, 280, 299, 271],
               [278, 282, 273, 288, 270, 265, 257, 302, 272, 274],
               [266, 292, 295, 287, 291, 264, 277, 279, 296, 297],
               [284, 289, 298, 285, 294, 260, 303, 255, 304, 290]]

for i in range(len(valid_study)):
    valid_list = []
    for val_id in valid_study[i]:
        current_list1 = [k for k in img_list if str(val_id) in k]
        valid_list += current_list1

    train_list = list(set(img_list).difference(set(valid_list)))
    write(train_list, 'train_new{}.txt'.format(i), data_dir=out_dir)
    write(valid_list, 'valid_new{}.txt'.format(i), data_dir=out_dir)
