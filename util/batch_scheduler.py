import os
import numpy as np
import random

batch = 16

def get_path(root, phase, index, img_list, modal):
    for r in root:
        tmp_path = os.path.join(r, phase, modal, img_list[index])
        if os.path.exists(tmp_path):
            return tmp_path
    return ''

root = ['/home1/yuezhang/data/TT/ISLES22',
        '/home1/yuezhang/data/TT/brats',
        '/home1/yuezhang/data/TT/glioma',
        '/home1/yuezhang/data/TT/immune'
        ]
modals = ['t1', 't2', 't1ce', 'flair', 'dwi', 'adc']
phase = 'train'
img_list = []
for r in root:
    for m in modals:
        if os.path.exists(os.path.join(r, phase, m)):
            img_list += os.listdir(os.path.join(r, phase, m))

img_set = set(img_list)
img_list = list(img_set)
random.shuffle(img_list)

# Groupping
file_dict = {}
for index in range(len(img_list)):
    gt_available_mask = np.ones((6))
    for i in range(6):
        if get_path(root, phase, index, img_list, modals[i])=='':
            gt_available_mask[i] = 0

    if str(gt_available_mask) not in file_dict.keys():
        file_dict[str(gt_available_mask)]=[]
    file_dict[str(gt_available_mask)].append(img_list[index])

# Padding
for key in file_dict.keys():
    list = file_dict[key]
    if len(list)%batch!=0:
        r = len(list)%batch
        addn = batch - r
        for a in range(addn):
            file_dict[key].append(file_dict[key][a])

f = open('/home1/yuezhang/code/UniSyn_git/util/file_list_b%s.txt'%(str(batch)), 'w')
for key in file_dict.keys():
    for name in file_dict[key]:
        f.write(name+'\n')
f.close()
print('done')