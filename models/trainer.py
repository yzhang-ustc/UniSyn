import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
# from models import network as networks
from models import network as networks
import numpy as np
import random
import torch.nn as nn

class UniSyn(BaseModel):
    def name(self):
        return 'UniSyn'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,opt.norm, not opt.no_dropout,
                                      opt.init_type, 0.02, self.gpu_ids, task_num=6)
        
#        self.vgg=VGG16().cuda()
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netDA = networks.define_D(opt.input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, 0.02, self.gpu_ids)
            self.netDB = networks.define_D(opt.input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, 0.02, self.gpu_ids)
            self.netDC = networks.define_D(opt.input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, 0.02, self.gpu_ids)
            self.netDD = networks.define_D(opt.input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, 0.02, self.gpu_ids)
            self.netDE = networks.define_D(opt.input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type,
                                           0.02, self.gpu_ids)
            self.netDF = networks.define_D(opt.input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type,
                                           0.02, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netDA, 'DA', opt.which_epoch)
                self.load_network(self.netDB, 'DB', opt.which_epoch)
                self.load_network(self.netDC, 'DC', opt.which_epoch)
                self.load_network(self.netDD, 'DD', opt.which_epoch)
                self.load_network(self.netDE, 'DE', opt.which_epoch)
                self.load_network(self.netDF, 'DF', opt.which_epoch)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DA = torch.optim.Adam(self.netDA.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DB = torch.optim.Adam(self.netDB.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DC = torch.optim.Adam(self.netDC.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DD = torch.optim.Adam(self.netDD.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DE = torch.optim.Adam(self.netDE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DF = torch.optim.Adam(self.netDF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_DA)
            self.optimizers.append(self.optimizer_DB)
            self.optimizers.append(self.optimizer_DC)
            self.optimizers.append(self.optimizer_DD)
            self.optimizers.append(self.optimizer_DE)
            self.optimizers.append(self.optimizer_DF)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        # networks.print_network(self.netG)
        print('-----------------------------------------------')

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def get_available_mask(self, epoch, gt_mask_tensor):
        available_mask = np.zeros((6))
        gt_mask = gt_mask_tensor.numpy()
        # gt_mask = gt_mask_tensor
        available_gt_num = np.sum(gt_mask)
        sub_mask = np.zeros((available_gt_num))
        random_drop_num = random.randint(1, available_gt_num-1)
        idxs = []
        for i in range(available_gt_num):
            idxs.append(i)
        for drop in range(random_drop_num):
            drop_idx = random.randint(0, len(idxs)-1)

            idxs.remove(idxs[drop_idx])
        for j in range(len(idxs)):
            sub_mask[idxs[j]]=1

        available_mask[gt_mask==1]=sub_mask
        available_mask = torch.LongTensor(available_mask).cuda()

        return available_mask

    def set_input(self, input, epoch):
        input_A = input['A']
        input_B = input['B']
        input_C = input['C']
        input_D = input['D']
        input_E = input['E']
        input_F = input['F']

        batch_size =input_A.size()[0]

        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0])
            input_B = input_B.cuda(self.gpu_ids[0])
            input_C = input_C.cuda(self.gpu_ids[0])
            input_D = input_D.cuda(self.gpu_ids[0])
            input_E = input_E.cuda(self.gpu_ids[0])
            input_F = input_F.cuda(self.gpu_ids[0])

        self.input_A = input_A
        self.input_B = input_B
        self.input_C = input_C
        self.input_D = input_D
        self.input_E = input_E
        self.input_F = input_F

        gt_mask = input['gt_mask'][0]
        self.input_mask = self.get_available_mask(epoch, gt_mask)
        self.gt_mask = input['gt_mask'][0].cuda()
        self.image_paths = input['A_paths']
        self.reverse_input_mask = (1 - self.input_mask)*(1-self.gt_mask)

        self.dataset = input['dataset'][0].cuda()
        # print(self.dataset)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_C = Variable(self.input_C)
        self.real_D = Variable(self.input_D)
        self.real_E = Variable(self.input_E)
        self.real_F = Variable(self.input_F)

        inputs = torch.cat([self.real_A, self.real_B, self.real_C,
                            self.real_D, self.real_E, self.real_F], dim=1)
        results = self.netG(inputs, self.dataset, self.input_mask)
        self.fake_A = results[0]
        self.fake_B = results[1]
        self.fake_C = results[2]
        self.fake_D = results[3]
        self.fake_E = results[4]
        self.fake_F = results[5]


    def test_single(self, available, target, available_id):
        # no backprop gradients
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)
        self.real_C = Variable(self.input_C, volatile=True)
        self.real_D = Variable(self.input_D, volatile=True)
        self.real_E = Variable(self.input_E, volatile=True)
        self.real_F = Variable(self.input_F, volatile=True)

        # A B C D -> 0 1 2 3
        if (self.gt_mask[available_id]):
            inputs = torch.cat([self.real_A, self.real_B, self.real_C,
                                self.real_D, self.real_E, self.real_F], dim=1)
            input_mask = torch.LongTensor(available).cuda()
            res = self.netG(inputs, self.dataset, input_mask)
            return res[target]
        return None

    def test_multi(self, available, target, available_id, available_num):
        # no backprop gradients
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)
        self.real_C = Variable(self.input_C, volatile=True)
        self.real_D = Variable(self.input_D, volatile=True)
        self.real_E = Variable(self.input_E, volatile=True)
        self.real_F = Variable(self.input_F, volatile=True)

        if available_num == 2:
            if (self.gt_mask[available_id[0]]) and (self.gt_mask[available_id[1]]):
                inputs = torch.cat([self.real_A, self.real_B, self.real_C,
                                    self.real_D, self.real_E, self.real_F], dim=1)
                input_mask = torch.LongTensor(available).cuda()
                res = self.netG(inputs, self.dataset, input_mask)
                return res[target]
        elif available_num == 3:
            if (self.gt_mask[available_id[0]]) and (self.gt_mask[available_id[1]])\
                    and (self.gt_mask[available_id[2]]):
                inputs = torch.cat([self.real_A, self.real_B, self.real_C,
                                    self.real_D, self.real_E, self.real_F], dim=1)
                input_mask = torch.LongTensor(available).cuda()
                res = self.netG(inputs, self.dataset, input_mask)
                return res[target]

        return None

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_input_mask(self):
        return self.input_mask
        
    def backward_DA(self):
        # Fake
        fake_A = self.netDA(self.fake_A.detach())
        self.loss_DA_fake = self.criterionGAN(fake_A, False)
        real_A = self.netDA(self.real_A)
        self.loss_DA_real = self.criterionGAN(real_A, True)
        self.loss_DA = (self.loss_DA_fake + self.loss_DA_real) * 0.5

        self.loss_DA.backward()
        
        

    def backward_DB(self):
        # Fake
        fake_B = self.netDB(self.fake_B.detach())
        self.loss_DB_fake = self.criterionGAN(fake_B, False)
        real_B = self.netDB(self.real_B)
        self.loss_DB_real = self.criterionGAN(real_B, True)
        self.loss_DB = (self.loss_DB_fake + self.loss_DB_real) * 0.5

        self.loss_DB.backward()
        
    def backward_DC(self):
        # Fake
        fake_C = self.netDC(self.fake_C.detach())

        self.loss_DC_fake = self.criterionGAN(fake_C, False)
        real_C = self.netDC(self.real_C)

        self.loss_DC_real = self.criterionGAN(real_C, True)
        self.loss_DC = (self.loss_DC_fake + self.loss_DC_real) * 0.5
        self.loss_DC.backward()
        
    def backward_DD(self):
        # Fake
        fake_D = self.netDD(self.fake_D.detach())

        self.loss_DD_fake = self.criterionGAN(fake_D, False)
        real_D = self.netDD(self.real_D)

        self.loss_DD_real = self.criterionGAN(real_D, True)
        self.loss_DD = (self.loss_DD_fake + self.loss_DD_real) * 0.5

        self.loss_DD.backward()

    def backward_DE(self):
        # Fake
        fake_E = self.netDE(self.fake_E.detach())

        self.loss_DE_fake = self.criterionGAN(fake_E, False)
        real_E = self.netDE(self.real_E)

        self.loss_DE_real = self.criterionGAN(real_E, True)
        self.loss_DE = (self.loss_DE_fake + self.loss_DE_real) * 0.5
        self.loss_DE.backward()

    def backward_DF(self):
        # Fake
        fake_F = self.netDF(self.fake_F.detach())

        self.loss_DF_fake = self.criterionGAN(fake_F, False)
        real_F = self.netDF(self.real_F)

        self.loss_DF_real = self.criterionGAN(real_F, True)
        self.loss_DF = (self.loss_DF_fake + self.loss_DF_real) * 0.5

        self.loss_DF.backward()

    def backward_G(self):
        # GAN loss
        self.loss_A_GAN = self.criterionGAN(self.netDA(self.fake_A), True)
        self.loss_B_GAN = self.criterionGAN(self.netDB(self.fake_B), True)
        self.loss_C_GAN = self.criterionGAN(self.netDC(self.fake_C), True)
        self.loss_D_GAN = self.criterionGAN(self.netDD(self.fake_D), True)
        self.loss_E_GAN = self.criterionGAN(self.netDE(self.fake_E), True)
        self.loss_F_GAN = self.criterionGAN(self.netDF(self.fake_F), True)
        # weight=1.0

        self.loss_G_GAN = self.loss_A_GAN*(1-self.input_mask[0])*self.gt_mask[0]\
                          + self.loss_B_GAN*(1-self.input_mask[1])*self.gt_mask[1]\
                          + self.loss_C_GAN*(1-self.input_mask[2])*self.gt_mask[2]\
                          + self.loss_D_GAN*(1-self.input_mask[3])*self.gt_mask[3]\
                          + self.loss_E_GAN * (1 - self.input_mask[4])*self.gt_mask[4]\
                          + self.loss_F_GAN * (1 - self.input_mask[5])*self.gt_mask[5]


        wA, wB, wC, wD, wE, wF = 1-self.input_mask[0], 1-self.input_mask[1], \
                         1-self.input_mask[2], 1-self.input_mask[3], \
                                 1 - self.input_mask[4], 1 - self.input_mask[5]
        if wA==0:
            wA = 0.3
        if wB==0:
            wB = 0.3
        if wC==0:
            wC = 0.3
        if wD==0:
            wD = 0.3
        if wE==0:
            wE = 0.3
        if wF==0:
            wF = 0.3
        self.loss_A_L1 = self.criterionL1(self.fake_A, self.real_A)
        self.loss_B_L1 = self.criterionL1(self.fake_B, self.real_B)
        self.loss_C_L1 = self.criterionL1(self.fake_C, self.real_C)
        self.loss_D_L1 = self.criterionL1(self.fake_D, self.real_D)
        self.loss_E_L1 = self.criterionL1(self.fake_E, self.real_E)
        self.loss_F_L1 = self.criterionL1(self.fake_F, self.real_F)

        self.loss_G_L1 = self.loss_A_L1*wA*self.gt_mask[0]  \
                         + self.loss_B_L1*wB*self.gt_mask[1]  \
                         + self.loss_C_L1*wC*self.gt_mask[2]  \
                         + self.loss_D_L1*wD*self.gt_mask[3]  \
                         + self.loss_E_L1 * wE*self.gt_mask[4]  \
                         + self.loss_F_L1 * wF*self.gt_mask[5]

        # total loss
        self.loss_G = self.loss_G_GAN + 100*self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.optimizer_G.zero_grad()
        self.forward()

        if (1 - self.input_mask[0]) and self.gt_mask[0]:
            self.optimizer_DA.zero_grad()
            self.backward_DA()
            self.optimizer_DA.step()

        if (1 - self.input_mask[1]) and self.gt_mask[1]:
            self.optimizer_DB.zero_grad()
            self.backward_DB()
            self.optimizer_DB.step()

        if (1 - self.input_mask[2]) and self.gt_mask[2]:
            self.optimizer_DC.zero_grad()
            self.backward_DC()
            self.optimizer_DC.step()

        if (1 - self.input_mask[3]) and self.gt_mask[3]:
            self.optimizer_DD.zero_grad()
            self.backward_DD()
            self.optimizer_DD.step()

        if (1 - self.input_mask[4]) and self.gt_mask[4]:
            self.optimizer_DE.zero_grad()
            self.backward_DE()
            self.optimizer_DE.step()

        if (1 - self.input_mask[5]) and self.gt_mask[5]:
            self.optimizer_DF.zero_grad()
            self.backward_DF()
            self.optimizer_DF.step()

        self.backward_G()
        self.optimizer_G.step()


    def get_current_errors(self):
        return OrderedDict([
                            ('A_L1', self.loss_A_L1.item()),
                            ('B_L1', self.loss_B_L1.item()),
                            ('C_L1', self.loss_C_L1.item()),
                            ('D_L1', self.loss_D_L1.item()),
                            ('E_L1', self.loss_E_L1.item()),
                            ('F_L1', self.loss_F_L1.item())
                            ])


    def get_current_visuals(self):
        real_A = self.real_A
        real_B = self.real_B
        real_C = self.real_C
        real_D = self.real_D
        real_E = self.real_E
        real_F = self.real_F

        fake_A = self.fake_A
        fake_B = self.fake_B
        fake_C = self.fake_C
        fake_D = self.fake_D
        fake_E = self.fake_E
        fake_F = self.fake_F

        return OrderedDict([('real_A', real_A), ('fake_A', fake_A),
                            ('real_B', real_B), ('fake_B', fake_B),
                            ('real_C', real_C), ('fake_C', fake_C),
                            ('real_D', real_D), ('fake_D', fake_D),
                            ('real_E', real_E), ('fake_E', fake_E),
                            ('real_F', real_F), ('fake_F', fake_F)
                            ])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netDA, 'DA', label, self.gpu_ids)
        self.save_network(self.netDB, 'DB', label, self.gpu_ids)
        self.save_network(self.netDC, 'DC', label, self.gpu_ids)
        self.save_network(self.netDD, 'DD', label, self.gpu_ids)
        self.save_network(self.netDE, 'DE', label, self.gpu_ids)
        self.save_network(self.netDF, 'DF', label, self.gpu_ids)



