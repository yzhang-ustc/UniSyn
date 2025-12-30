import os
import numpy as np
import torch
import torchvision.utils as vutils
import PIL.Image as Image
from torchvision.transforms import transforms
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import cv2

def test():
    import sys
    import numpy as np
    sys.argv = args
    import os
    from options.test_options import TestOptions
    from test_brats.data__init__ import create_dataset as create_dataset_brats
    from models.modal__init__ import create_model
    from util.visualizer import Visualizer

    opt = TestOptions().parse()
    opt.num_workers = 1  # test code only supports nThreads = 1
    opt.batch_size = 1  # test code only supports batchSize = 1
    opt.unshuffle = True  # no shuffle

    print(opt)
    data_loader = create_dataset_brats(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)

    result_dir_1 = os.path.join(opt.results_dir, opt.name, 'brats_11_t1t1c_{}'.format(opt.which_epoch))
    os.makedirs(result_dir_1, exist_ok=True)
    result_path_1 = os.path.join(result_dir_1, 'images')
    os.makedirs(os.path.join(result_path_1, 't1c'), exist_ok=True)

    result_dir_2 = os.path.join(opt.results_dir, opt.name, 'brats_11_t2flair_{}'.format(opt.which_epoch))
    os.makedirs(result_dir_2, exist_ok=True)
    result_path_2 = os.path.join(result_dir_2, 'images')
    os.makedirs(os.path.join(result_path_2, 'flair'), exist_ok=True)

    for i, data in enumerate(dataset):
        # if i>200:
        #     break
        print('brats:', i)
        model.set_input(data, 0)  # unpack data from data loader

        outputs = model.test_single([1, 0, 0, 0, 0, 0], 2, 0)
        outputs_2 = model.test_single([0, 1, 0, 0, 0, 0], 3, 1)
        if (outputs != None):
            s_t1c = outputs  # run inference
            s_t1c = from_neg_to_img(s_t1c)
            img_path = model.get_image_paths()  # get image paths
            vutils.save_image(s_t1c, os.path.join(result_path_1, 't1c',
                                                 img_path[0].split('/')[-1]))
        if (outputs_2 != None):
            s_flair = outputs_2  # run inference
            s_flair = from_neg_to_img(s_flair)
            img_path = model.get_image_paths()  # get image paths
            vutils.save_image(s_flair, os.path.join(result_path_2, 'flair',
                                                    img_path[0].split('/')[-1]))


def from_neg_to_img(img):
    return (img + 1) / 2.


import sys

sys.argv.extend(['--model', 'pGAN'])
args = sys.argv
sys.argv.extend(['--unshuffle'])
test()
