import os
import torch
from torch.utils.data import Dataset
import glob
from torchvision.transforms import transforms
import numpy as np
import PIL.Image as Image
from PIL import ImageEnhance, ImageOps, ImageFile, ImageFilter
import random


class BratsDataset(Dataset):
    def __init__(self, opt, transform):
        super(BratsDataset, self).__init__()

        self.root = [
            # '/home1/yuezhang/data/TT/ISLES22',
            '/home1/yuezhang/data/TT/brats',
            # '/home1/yuezhang/data/TT/glioma',
            # '/home1/yuezhang/data/TT/immune'
        ]
        self.modals = ['t1', 't2', 't1ce', 'flair', 'dwi', 'adc']
        self.phase = opt.phase
        self.img_list = []
        img_list = []
        ##############################
        for r in self.root:
            for m in self.modals:
                if os.path.exists(os.path.join(r, self.phase, m)):
                    img_list += os.listdir(os.path.join(r, self.phase, m))

        img_set = set(img_list)
        self.img_list = list(img_set)
        self.img_list.sort()
        # random.shuffle(self.img_list)

        ###############################

        self.transform = transform
        self.opt = opt

    def augmentation(self, imgA, imgB, imgC, imgD, imgE, imgF):
        if self.opt.augmentation == True:
            nflip = random.randint(0, 1)
            nblur = random.randint(0, 1)
            nrotate = random.randint(0, 1)
            if nflip:
                imgA = imgA.transpose(Image.FLIP_LEFT_RIGHT)
                imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)
                imgC = imgC.transpose(Image.FLIP_LEFT_RIGHT)
                imgD = imgD.transpose(Image.FLIP_LEFT_RIGHT)
                imgE = imgE.transpose(Image.FLIP_LEFT_RIGHT)
                imgF = imgF.transpose(Image.FLIP_LEFT_RIGHT)
            if nrotate:
                random_angle = np.random.randint(-10, 10)
                imgA = imgA.rotate(random_angle, Image.BICUBIC)
                imgB = imgB.rotate(random_angle, Image.BICUBIC)
                imgC = imgC.rotate(random_angle, Image.BICUBIC)
                imgD = imgD.rotate(random_angle, Image.BICUBIC)
                imgE = imgE.rotate(random_angle, Image.BICUBIC)
                imgF = imgF.rotate(random_angle, Image.BICUBIC)

        return imgA, imgB, imgC, imgD, imgE, imgF

    def norm_neg_one_to_one(self, img):
        return img * 2 - 1

    def get_path(self, index, modal):
        for r in self.root:
            tmp_path = os.path.join(r, self.phase, modal, self.img_list[index])
            if os.path.exists(tmp_path):
                return tmp_path
        return ''

    def get_img(self, index, modal):
        path = self.get_path(index, modal)
        # print(path)
        if path != '':
            return Image.open(path)
        return Image.new('L', (192, 192), 0)

    def __getitem__(self, index):
        path = self.get_path(index, 't1')
        if path == '':
            path = self.get_path(index, 't2')
        if path == '':
            path = self.get_path(index, 't1ce')
        if path == '':
            path = self.get_path(index, 'flair')
        if path == '':
            path = self.get_path(index, 'dwi')
        if path == '':
            path = self.get_path(index, 'adc')
        
        dataset = torch.LongTensor([1])

        gt_available_mask = np.ones((6))
        for i in range(6):
            if self.get_path(index, self.modals[i]) == '':
                gt_available_mask[i] = 0
        gt_available_mask = torch.LongTensor(gt_available_mask)

        A_img = self.get_img(index, 't1')
        B_img = self.get_img(index, 't2')
        C_img = self.get_img(index, 't1ce')
        D_img = self.get_img(index, 'flair')
        E_img = self.get_img(index, 'dwi')
        F_img = self.get_img(index, 'adc')
        # print(A_path)

        A_img, B_img, C_img, D_img, E_img, F_img = self.augmentation(A_img, B_img, C_img, D_img, E_img, F_img)

        A = self.transform(A_img)
        A = A.float()
        A = self.norm_neg_one_to_one(A)

        B = self.transform(B_img)
        B = B.float()
        B = self.norm_neg_one_to_one(B)

        C = self.transform(C_img)
        C = C.float()
        C = self.norm_neg_one_to_one(C)

        D = self.transform(D_img)
        D = D.float()
        D = self.norm_neg_one_to_one(D)

        E = self.transform(E_img)
        E = E.float()
        E = self.norm_neg_one_to_one(E)

        F = self.transform(F_img)
        F = F.float()
        F = self.norm_neg_one_to_one(F)

        return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F,
                'A_paths': path, 'gt_mask': gt_available_mask, 'dataset': dataset}

    def __len__(self):
        return len(self.img_list)
