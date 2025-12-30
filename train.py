import os
import numpy as np
import torch
import torchvision.utils as vutils
import PIL.Image as Image
from torchvision.transforms import transforms
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import cv2

def train():
    import time
    from options.train_options import TrainOptions
    from models.data__init__ import create_dataset
    from models.modal__init__ import create_model
    from util.visualizer import Visualizer

    opt = TrainOptions().parse()
    model = create_model(opt)
    # Loading data
    data_loader = create_dataset(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('Training images = %d' % dataset_size)
    visualizer = Visualizer(opt)
    total_steps = (opt.epoch_count) * 10000
    # Starts training

    # opt.phase = 'train'
    # opt.unshuffle = False  # no shuffle
    # opt.augmentation = True

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        epoch_loss = {}

        for i, data in enumerate(dataset):
            step = i + 1
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data, epoch)
            model.optimize_parameters()

            errors = model.get_current_errors()
            for k, v in errors.items():
                if i == 0:
                    epoch_loss[k] = v
                else:
                    epoch_loss[k] += v

            if total_steps % opt.print_freq == 0:
                save_loss = False
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data, save_loss)

            # Save model based on the number of iterations
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

                visuals = model.get_current_visuals()  # get image results
                image_outputs = get_train_output(visuals)
                all_images = torch.cat(image_outputs, dim=0)
                img_path = model.get_image_paths()
                available_mask = model.get_input_mask()
                available_mask = available_mask.cpu().numpy()
                if not os.path.exists(opt.checkpoints_dir + '/adjust_lambda/train_images'):
                    os.makedirs(opt.checkpoints_dir + '/adjust_lambda/train_images')

                vutils.save_image(all_images,
                                  opt.checkpoints_dir + '/adjust_lambda/train_images/[%s]%s%s' % (total_steps, str(available_mask), img_path[0].split('/')[-1]),
                                  nrow=opt.batch_size)
                torch.cuda.empty_cache()

            iter_data_time = time.time()

        for k, v in epoch_loss.items():
            epoch_loss[k] = v / step
        save_loss = True
        t = (time.time() - iter_start_time) / opt.batch_size
        sssss = 00000
        visualizer.print_current_errors(epoch, sssss, errors, t, t_data, save_loss)

        # Save model based on the number of epochs
        print(opt.dataset_mode)
        if epoch % opt.save_epoch_freq == 0 and epoch >= 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save('continue_datasetemb_'+str(epoch))

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()


def get_train_output(visuals):

        return from_neg_to_img(visuals['real_A']), \
               from_neg_to_img(visuals['fake_A']), \
               from_neg_to_img(visuals['real_B']), \
               from_neg_to_img(visuals['fake_B']), \
               from_neg_to_img(visuals['real_C']), \
               from_neg_to_img(visuals['fake_C']), \
               from_neg_to_img(visuals['real_D']), \
               from_neg_to_img(visuals['fake_D']), \
               from_neg_to_img(visuals['real_E']), \
               from_neg_to_img(visuals['fake_E']), \
               from_neg_to_img(visuals['real_F']), \
               from_neg_to_img(visuals['fake_F'])


def from_neg_to_img(img):
    return (img + 1) / 2.


import sys

sys.argv.extend(['--model', 'pGAN'])
args = sys.argv
if '--training' in str(args):
    train()
