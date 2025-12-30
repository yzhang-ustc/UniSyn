import numpy as np
import os
import ntpath
import time
from . import util
from . import html
#from scipy.misc import imresize
import math
import numpy
from scipy.ndimage import gaussian_filter
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

class Visualizer():
    def __init__(self, opt):
        self.name = opt.name
        self.opt = opt
        self.saved = False
        self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
        if not opt.isTrain:
            self.result_dir = os.path.join(opt.results_dir, opt.name)
            util.mkdirs([self.result_dir])
        else:
            util.mkdirs([self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch,iteration, save_result):
        self.saved = True
        for label, image_numpy in visuals.items():
            img_path = os.path.join(self.img_dir, 'epoch%.3d_%.3d_%s.png' % (epoch,iteration, label))
            util.save_image(image_numpy, img_path)
                
    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t, t_data, save_loss):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        
        if save_loss:
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)
            
    # save image to the disk
    def save_images_calculation_index(self, result_dir, visuals, image_path):
        image_dir = os.path.join(result_dir, 'images')
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
            os.makedirs(image_dir + '/AB')
            os.makedirs(image_dir + '/AC')
            os.makedirs(image_dir + '/AD')
            os.makedirs(image_dir + '/BA')
            os.makedirs(image_dir + '/BC')
            os.makedirs(image_dir + '/BD')
            os.makedirs(image_dir + '/CA')
            os.makedirs(image_dir + '/CB')
            os.makedirs(image_dir + '/CD')
            os.makedirs(image_dir + '/DA')
            os.makedirs(image_dir + '/DB')
            os.makedirs(image_dir + '/DC')
        
        ret = {}
        for label, im in visuals.items():
            image_tensor = im.data
            image_numpy = image_tensor[0].cpu().float().numpy()
            image_numpy = (np.transpose(image_numpy, (0, 1, 2)) + 1) / 2.0 *255
            image_numpy = np.clip(image_numpy, 0, 255)
            # image_numpy = np.uint8((np.transpose(image_numpy, (0, 1, 2)) + 1) / 2.0 * 255)
            ret[label] = image_numpy
            path = name.split('_')[-2]
            
        gtA = ret['real_A']
        gtB = ret['real_B']
        gtC = ret['real_C']
        gtD = ret['real_D']
        
        outAB = ret['fake_AB']
        outAC = ret['fake_AC']
        outAD = ret['fake_AD']
        outBA = ret['fake_BA']
        outBC = ret['fake_BC']
        outBD = ret['fake_BD']
        outCA = ret['fake_CA']
        outCB = ret['fake_CB']
        outCD = ret['fake_CD']
        outDA = ret['fake_DA']
        outDB = ret['fake_DB']
        outDC = ret['fake_DC']
        
        OUT = [ outAB[0,:,:],outAC[0,:,:],outAD[0,:,:],
                outBA[0,:,:],outBC[0,:,:],outBD[0,:,:],
                outCA[0,:,:],outCB[0,:,:],outCD[0,:,:],
                outDA[0,:,:],outDB[0,:,:],outDC[0,:,:]
                ]
        
        
        MAE_INDEX = [mae(gtB,outAB),mae(gtC,outAC),mae(gtD,outAD),mae(gtA,outBA),mae(gtC,outBC),mae(gtD,outBD),mae(gtA,outCA),mae(gtB,outCB),mae(gtD,outCD),mae(gtA,outDA),mae(gtB,outDB),mae(gtC,outDC)]
        PSNR_INDEX = [psnr(gtB,outAB),psnr(gtC,outAC),psnr(gtD,outAD),psnr(gtA,outBA),psnr(gtC,outBC),psnr(gtD,outBD),psnr(gtA,outCA),psnr(gtB,outCB),psnr(gtD,outCD),psnr(gtA,outDA),psnr(gtB,outDB),psnr(gtC,outDC)]
        SSIM_INDEX = [ssim(gtB,outAB),ssim(gtC,outAC),ssim(gtD,outAD),ssim(gtA,outBA),ssim(gtC,outBC),ssim(gtD,outBD),ssim(gtA,outCA),ssim(gtB,outCB),ssim(gtD,outCD),ssim(gtA,outDA),ssim(gtB,outDB),ssim(gtC,outDC)]
        # print(MAE_INDEX)
        # print(PSNR_INDEX)
        # print(SSIM_INDEX)
        
        return image_dir, path, MAE_INDEX, PSNR_INDEX, SSIM_INDEX, OUT
        
        
def mae(img1, img2):
    return numpy.mean(abs(img1 - img2))

def psnr(img1, img2):
    # mse = numpy.mean( (img1 - img2) ** 2 )
    # if mse == 0:
    #     return 100
    # PIXEL_MAX = 255.0
    # return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return compare_psnr(img1[0,:,:],img2[0,:,:], data_range=255)

def ssim(img1,img2):
    # print(img1.shape)
    score=compare_ssim(img1[0,:,:],img2[0,:,:],data_range=255)
    return score


