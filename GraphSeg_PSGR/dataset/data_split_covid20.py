import sys
sys.path.append('../')
from os import listdir
from os.path import isfile, join


def write(data, fname, data_dir=None):
    fname = join(data_dir, fname)
    with open(fname, 'w') as f:
        f.write('\n'.join(data))


input_dir = '/media/userdisk0/hzjia/Data/covid20/slices/masks'
out_dir = '/media/userdisk0/hzjia/Data/covid20/slices'

img_list = []
for img in listdir(input_dir):
    if isfile(join(input_dir, img)) and 'npy' in img:
        img_list.append(img)

write(img_list, 'all_list_new.txt', out_dir)

valid_study = [['coronacases_007', 'coronacases_010', 'coronacases_006', 'radiopaedia_29_86491_1'],
               ['coronacases_008', 'radiopaedia_10_85902_3', 'radiopaedia_4_85506_1', 'radiopaedia_27_86410_0'],
               ['coronacases_009', 'coronacases_005', 'coronacases_001', 'radiopaedia_14_85914_0'],
               ['radiopaedia_36_86526_0', 'coronacases_003', 'radiopaedia_29_86490_1', 'radiopaedia_10_85902_1'],
               ['coronacases_002', 'radiopaedia_7_85703_0', 'radiopaedia_40_86625_0', 'coronacases_004']]

for i in range(len(valid_study)):
    valid_list = []
    for val_id in valid_study[i]:
        current_list1 = [k for k in img_list if str(val_id) in k]
        valid_list += current_list1
    valid_list = list(set(valid_list))
    train_list = list(set(img_list).difference(set(valid_list)))
    write(train_list, 'train_new{}.txt'.format(i), data_dir=out_dir)
    write(valid_list, 'valid_new{}.txt'.format(i), data_dir=out_dir)
