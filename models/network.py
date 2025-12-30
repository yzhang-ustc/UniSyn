import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import math
from einops import rearrange

###############################################################################
# Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    # Learning rate policies
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        # import BalancedDataParallel
        # if len(gpu_ids)>1:
        # net = BalancedDataParallel(12, net, dim=0)
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], task_num=6):
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    if use_gpu:
        assert (torch.cuda.is_available())

    netG = Generator(1, 1, task_num)
    return init_net(netG, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)
    if use_gpu:
        assert (torch.cuda.is_available())
    netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                               gpu_ids=gpu_ids)
    return init_net(netD, init_type, init_gain, gpu_ids)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        # lsGAN or vanillaGAN
        if use_lsgan:
            self.loss = nn.MSELoss()
            # print('lsgan')
        else:
            self.loss = nn.BCELoss()
            # print('bcegan')

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)



# Defines a layer discriminator.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_sigmoid = use_sigmoid
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        # self.conv1 = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        # self.conv2 = nn.Conv2d(ndf * nf_mult, 5, kernel_size=23, bias=False)

    def forward(self, input):

        return self.model(input)


def exists(x):
    return x is not None

class DoubleConv_enc(nn.Module):
    def __init__(self, nb_tasks, in_ch, out_ch, in_ch_c, d_emb_dim):
        super(DoubleConv_enc, self).__init__()
        # self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.common_conv_1 = nn.Conv2d(in_ch_c, out_ch, 3, padding=1)
        self.common_conv_2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dwch_conv1 = nn.ModuleList([nn.Conv2d(out_ch * 2, out_ch, 1, padding=0) for i in range(nb_tasks)])
        self.dwch_conv2 = nn.ModuleList([nn.Conv2d(out_ch * 2, out_ch, 1, padding=0) for i in range(nb_tasks)])

        self.conv1 = nn.ModuleList([nn.Conv2d(in_ch, out_ch, 3, padding=1) for i in range(nb_tasks)])
        self.conv2 = nn.ModuleList([nn.Conv2d(out_ch, out_ch, 3, padding=1) for i in range(nb_tasks)])

        self.norm1 = nn.ModuleList([nn.InstanceNorm2d(out_ch) for i in range(nb_tasks)])
        self.relu1 = nn.ReLU(inplace=True)

        self.norm2 = nn.ModuleList([nn.InstanceNorm2d(out_ch) for i in range(nb_tasks)])
        self.relu2 = nn.ReLU(inplace=True)


        self.common_norm1 = nn.InstanceNorm2d(out_ch)
        self.common_relu1 = nn.ReLU(inplace=True)
        self.common_norm2 = nn.InstanceNorm2d(out_ch)
        self.common_relu2 = nn.ReLU(inplace=True)

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_emb_dim, out_ch * 2)
        )

    def forward(self, input, c_input, id, d_emb = None):
        scale_shift = None
        if exists(self.mlp) and exists(d_emb):
            d_emb = self.mlp(d_emb)
            d_emb = rearrange(d_emb, 'b c -> b c 1 1')
            scale_shift = d_emb.chunk(2, dim=1)
        c_out1 = self.common_conv_1(c_input)
        out1 = self.conv1[id](input)
        out1 = self.dwch_conv1[id](torch.cat([out1, c_out1], dim=1))
        out1 = self.norm1[id](out1)
        out1 = self.relu1(out1)

        # c_out1 = self.common_in_relu1(c_out1)
        c_out1 = self.common_norm1(c_out1)
        c_out1 = self.common_relu1(c_out1)
        c_out2 = self.common_conv_2(c_out1)

        out2 = self.conv2[id](out1)
        out2 = self.dwch_conv2[id](torch.cat([out2, c_out2], dim=1))
        out2 = self.norm2[id](out2)
        if exists(scale_shift):
            scale, shift = scale_shift
            out2 = out2 * (scale + 1) + shift

        out2 = self.relu2(out2)
        # c_out2 = self.common_in_relu2(c_out2)
        c_out2 = self.common_norm2(c_out2)
        if exists(scale_shift):
            scale, shift = scale_shift
            c_out2 = c_out2 * (scale + 1) + shift

        c_out2 = self.common_relu2(c_out2)


        return out2, c_out2

class DoubleConv_share(nn.Module):
    def __init__(self, in_ch, out_ch, d_emb_dim):
        super(DoubleConv_share, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_emb_dim, out_ch * 2)
        )

    def forward(self, input, d_emb = None):
        scale_shift = None
        if exists(self.mlp) and exists(d_emb):
            d_emb = self.mlp(d_emb)
            d_emb = rearrange(d_emb, 'b c -> b c 1 1')
            scale_shift = d_emb.chunk(2, dim=1)
        out1 = self.conv1(input)
        out1 = self.norm1(out1)
        out1 = self.relu1(out1)

        out2 = self.conv2(out1)
        out2 = self.norm2(out2)
        if exists(scale_shift):
            scale, shift = scale_shift
            out2 = out2 * (scale + 1) + shift
        out2 = self.relu2(out2)

        return out2

class DoubleConv_dec(nn.Module):
    def __init__(self, nb_tasks, in_ch, out_ch, d_emb_dim):
        super(DoubleConv_dec, self).__init__()
        # self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv1 = nn.ModuleList([nn.Conv2d(in_ch, out_ch, 3, padding=1) for i in range(nb_tasks)])
        self.conv2 = nn.ModuleList([nn.Conv2d(out_ch, out_ch, 3, padding=1) for i in range(nb_tasks)])

        self.norm1 = nn.ModuleList([nn.InstanceNorm2d(out_ch) for i in range(nb_tasks)])
        self.relu1 = nn.ReLU(inplace=True)

        self.norm2 = nn.ModuleList([nn.InstanceNorm2d(out_ch) for i in range(nb_tasks)])
        self.relu2 = nn.ReLU(inplace=True)

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_emb_dim, out_ch * 2)
        )

    def forward(self, input, id, d_emb = None):
        scale_shift = None
        if exists(self.mlp) and exists(d_emb):
            d_emb = self.mlp(d_emb)
            d_emb = rearrange(d_emb, 'b c -> b c 1 1')
            scale_shift = d_emb.chunk(2, dim=1)
        out1 = self.conv1[id](input)
        out1 = self.norm1[id](out1)
        out1 = self.relu1(out1)


        out2 = self.conv2[id](out1)
        out2 = self.norm2[id](out2)
        if exists(scale_shift):
            scale, shift = scale_shift
            out2 = out2 * (scale + 1) + shift
        out2 = self.relu2(out2)

        return out2

class Att_block(nn.Module):
    def __init__(self, in_ch):
        super(Att_block, self).__init__()
        self.conv_scale_1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv_scale_2 = nn.Conv2d(in_ch, 32, 5, padding=2)
        self.conv_scale_3 = nn.Conv2d(in_ch, 32, 7, padding=3)
        self.final = nn.Sequential(nn.Conv2d(3*32, 1, 1),
                                   nn.Sigmoid())

    def forward(self, x):
        x_reduce = x
        x_scale_1 = self.conv_scale_1(x_reduce)
        x_scale_2 = self.conv_scale_2(x_reduce)
        x_scale_3 = self.conv_scale_3(x_reduce)
        att = self.final(torch.cat([x_scale_1, x_scale_2, x_scale_3], dim=1))
        return att

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, nb_tasks):
        super(Generator, self).__init__()
        ch = [64, 128, 256, 512, 1024]
        dim = 64
        fourier_dim = dim
        dataset_dim = dim * 4
        sinu_pos_emb = SinusoidalPosEmb(dim)
        self.dataset_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, dataset_dim),
            nn.GELU(),
            nn.Linear(dataset_dim, dataset_dim)
        )
        self.conv1 = DoubleConv_enc(nb_tasks, in_ch, ch[0], in_ch, dataset_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv_enc(nb_tasks, ch[0], ch[1] ,ch[0], dataset_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv_enc(nb_tasks, ch[1], ch[2], ch[1], dataset_dim)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv_enc(nb_tasks, ch[2], ch[3], ch[2], dataset_dim)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv_share(ch[3], ch[4], dataset_dim)
        # self.up6 = nn.ConvTranspose2d(ch[4], ch[3], 2, stride=2)
        self.up6 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                 nn.Conv2d(ch[4], ch[3], 1)
        )
        self.conv6 = DoubleConv_share(ch[4], ch[3], dataset_dim)
        # self.up7 = nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2)
        self.up7 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                 nn.Conv2d(ch[3], ch[2], 1)
                                 )
        self.conv7 = DoubleConv_dec(nb_tasks, ch[3], ch[2], dataset_dim)
        # self.up8 = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2)
        self.up8 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                 nn.Conv2d(ch[2], ch[1], 1)
                                 )
        self.conv8 = DoubleConv_dec(nb_tasks, ch[2], ch[1], dataset_dim)
        # self.up9 = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2)
        self.up9 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                 nn.Conv2d(ch[1], ch[0], 1)
                                 )
        self.conv9 = DoubleConv_dec(nb_tasks, ch[1], ch[0], dataset_dim)
        self.conv10 = nn.ModuleList([nn.Conv2d(ch[0], out_ch, 1) for i in range(nb_tasks)])

        self.fuse_1_A = Att_block(64)
        self.fuse_1_B = Att_block(64)
        self.fuse_1_C = Att_block(64)
        self.fuse_1_D = Att_block(64)
        self.fuse_1_E = Att_block(64)
        self.fuse_1_F = Att_block(64)
        self.fuse_2_A =  Att_block(128)
        self.fuse_2_B = Att_block(128)
        self.fuse_2_C = Att_block(128)
        self.fuse_2_D = Att_block(128)
        self.fuse_2_E = Att_block(128)
        self.fuse_2_F = Att_block(128)
        self.fuse_3_A =  Att_block(256)
        self.fuse_3_B = Att_block(256)
        self.fuse_3_C = Att_block(256)
        self.fuse_3_D = Att_block(256)
        self.fuse_3_E = Att_block(256)
        self.fuse_3_F = Att_block(256)
        self.fuse_4_A =  Att_block(512)
        self.fuse_4_B = Att_block(512)
        self.fuse_4_C = Att_block(512)
        self.fuse_4_D = Att_block(512)
        self.fuse_4_E = Att_block(512)
        self.fuse_4_F = Att_block(512)
        self.fuse_5_A =  Att_block(1024)
        self.fuse_5_B = Att_block(1024)
        self.fuse_5_C = Att_block(1024)
        self.fuse_5_D = Att_block(1024)
        self.fuse_5_E = Att_block(1024)
        self.fuse_5_F = Att_block(1024)

        self.ajust_dim_1 = nn.Sequential(nn.Conv2d(64*2, 64, 1),
                                         nn.InstanceNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.ajust_dim_2 = nn.Sequential(nn.Conv2d(128 * 2, 128, 1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.ajust_dim_3 = nn.Sequential(nn.Conv2d(256 * 2, 256, 1),
                                         nn.InstanceNorm2d(256),
                                         nn.ReLU(inplace=True))
        self.ajust_dim_4 = nn.Sequential(nn.Conv2d(512 * 2, 512, 1),
                                         nn.InstanceNorm2d(512),
                                         nn.ReLU(inplace=True))
        self.ajust_dim_5 = nn.Sequential(nn.Conv2d(1024 * 2, 1024, 1),
                                         nn.InstanceNorm2d(1024),
                                         nn.ReLU(inplace=True))


    def forward(self, x, dataset=None, input_ids=[]):
        d_emb = None
        if dataset != None:
            d_emb = self.dataset_mlp(dataset)
            # print(d_emb.size())
        input_A, input_B, input_C, input_D, input_E, input_F = x[:,0:1,:,:]*input_ids[0], x[:,1:2,:,:]*input_ids[1], \
                                             x[:,2:3,:,:]*input_ids[2], x[:,3:4,:,:]*input_ids[3], \
                                             x[:, 4:5, :, :] * input_ids[4], x[:, 5:6, :, :] * input_ids[5]

        c1_A, c2_A, c3_A, c4_A, c5_A = self.encoder(input_A, input_A, 0, d_emb)
        c1_B, c2_B, c3_B, c4_B, c5_B = self.encoder(input_B, input_B, 1, d_emb)
        c1_C, c2_C, c3_C, c4_C, c5_C = self.encoder(input_C, input_C, 2, d_emb)
        c1_D, c2_D, c3_D, c4_D, c5_D = self.encoder(input_D, input_D, 3, d_emb)
        c1_E, c2_E, c3_E, c4_E, c5_E = self.encoder(input_E, input_E, 4, d_emb)
        c1_F, c2_F, c3_F, c4_F, c5_F = self.encoder(input_F, input_F, 5, d_emb)

        #get unified multi-scale features c1-c5
        c1_maps_A, c1_maps_B, c1_maps_C, c1_maps_D, c1_maps_E, c1_maps_F = self.fuse_1_A(c1_A), \
                                                     self.fuse_1_B(c1_B), \
                                                     self.fuse_1_C(c1_C), \
                                                     self.fuse_1_D(c1_D), \
                                                     self.fuse_1_E(c1_E), \
                                                     self.fuse_1_F(c1_F)
        c1 = self.ajust_dim_1(self.get_fused_feature(c1_A, c1_B, c1_C, c1_D, c1_E, c1_F,
                                                     c1_maps_A, c1_maps_B, c1_maps_C, c1_maps_D,
                                                     c1_maps_E, c1_maps_F, input_ids))


        c2_maps_A, c2_maps_B, c2_maps_C, c2_maps_D, c2_maps_E, c2_maps_F = self.fuse_2_A(c2_A), \
                                                     self.fuse_2_B(c2_B), \
                                                     self.fuse_2_C(c2_C), \
                                                     self.fuse_2_D(c2_D), \
                                                     self.fuse_2_E(c2_E), \
                                                     self.fuse_2_F(c2_F)
        c2 = self.ajust_dim_2(self.get_fused_feature(c2_A, c2_B, c2_C, c2_D, c2_E, c2_F,
                                                     c2_maps_A, c2_maps_B, c2_maps_C, c2_maps_D,
                                                     c2_maps_E, c2_maps_F, input_ids))

        c3_maps_A, c3_maps_B, c3_maps_C, c3_maps_D, c3_maps_E, c3_maps_F = self.fuse_3_A(c3_A), \
                                                     self.fuse_3_B(c3_B), \
                                                     self.fuse_3_C(c3_C), \
                                                     self.fuse_3_D(c3_D), \
                                                     self.fuse_3_E(c3_E), \
                                                     self.fuse_3_F(c3_F)
        c3 = self.ajust_dim_3(self.get_fused_feature(c3_A, c3_B, c3_C, c3_D, c3_E, c3_F,
                                                     c3_maps_A, c3_maps_B, c3_maps_C, c3_maps_D,
                                                     c3_maps_E, c3_maps_F, input_ids))

        c4_maps_A, c4_maps_B, c4_maps_C, c4_maps_D, c4_maps_E, c4_maps_F = self.fuse_4_A(c4_A), \
                                                     self.fuse_4_B(c4_B), \
                                                     self.fuse_4_C(c4_C), \
                                                     self.fuse_4_D(c4_D),\
                                                     self.fuse_4_E(c4_E), \
                                                     self.fuse_4_F(c4_F),
        c4 = self.ajust_dim_4(self.get_fused_feature(c4_A, c4_B, c4_C, c4_D, c4_E, c4_F,
                                                     c4_maps_A, c4_maps_B, c4_maps_C, c4_maps_D,
                                                     c4_maps_E, c4_maps_F,input_ids))


        c5_maps_A, c5_maps_B, c5_maps_C, c5_maps_D, c5_maps_E, c5_maps_F = self.fuse_5_A(c5_A), \
                                                     self.fuse_5_B(c5_B), \
                                                     self.fuse_5_C(c5_C), \
                                                     self.fuse_5_D(c5_D),\
                                                     self.fuse_5_E(c5_E), \
                                                     self.fuse_5_F(c5_F),
        c5 = self.ajust_dim_5(self.get_fused_feature(c5_A, c5_B, c5_C, c5_D, c5_E, c5_F,
                                                     c5_maps_A, c5_maps_B, c5_maps_C, c5_maps_D,
                                                     c5_maps_E, c5_maps_F, input_ids))

        syn_A = self.decoder(c1, c2, c3, c4, c5, 0, d_emb)
        syn_B = self.decoder(c1, c2, c3, c4, c5, 1, d_emb)
        syn_C = self.decoder(c1, c2, c3, c4, c5, 2, d_emb)
        syn_D = self.decoder(c1, c2, c3, c4, c5, 3, d_emb)
        syn_E = self.decoder(c1, c2, c3, c4, c5, 4, d_emb)
        syn_F = self.decoder(c1, c2, c3, c4, c5, 5, d_emb)
        return syn_A, syn_B, syn_C, syn_D, syn_E, syn_F

    def decoder(self, c1, c2, c3, c4, c5, output_id, d_emb):
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6, d_emb)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7, output_id, d_emb)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8, output_id, d_emb)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9, output_id, d_emb)
        c10 = self.conv10[output_id](c9)
        syn = nn.Tanh()(c10)
        return syn


    def get_fused_feature(self, c1_A, c1_B, c1_C, c1_D, c1_E, c1_F,
                          c1_maps_A, c1_maps_B, c1_maps_C,
                          c1_maps_D, c1_maps_E, c1_maps_F, input_ids):

        c1_list = []
        c1_maps_list = []
        if input_ids[0]:
            c1_list.append(c1_A)
            c1_maps_list.append(c1_maps_A)
        if input_ids[1]:
            c1_list.append(c1_B)
            c1_maps_list.append(c1_maps_B)
        if input_ids[2]:
            c1_list.append(c1_C)
            c1_maps_list.append(c1_maps_C)
        if input_ids[3]:
            c1_list.append(c1_D)
            c1_maps_list.append(c1_maps_D)
        if input_ids[4]:
            c1_list.append(c1_E)
            c1_maps_list.append(c1_maps_E)
        if input_ids[5]:
            c1_list.append(c1_F)
            c1_maps_list.append(c1_maps_F)
        if len(c1_maps_list)>1:
            c1_maps_norm = nn.Softmax(dim=1)(torch.cat(c1_maps_list, dim=1))
            # c1_maps_norm = torch.cat(c1_maps_list, dim=1)
            c1_fea = torch.cat(c1_list, dim=1)
            c1_fea_list_new = []
            for f, map in zip(c1_fea.chunk(len(c1_maps_list), dim=1), c1_maps_norm.chunk(len(c1_maps_list), dim=1)):
                c1_fea_list_new.append(f.mul(map).unsqueeze(0))
            c1 = torch.cat(c1_fea_list_new, dim=0)
            c1 = torch.sum(c1, dim=0)
        else:
            c1 = c1_list[0].mul(c1_maps_list[0])

        c_max = torch.max(torch.stack(c1_list, dim=0), dim=0)[0]
        fused = torch.cat([c1, c_max], dim=1)
        # print(fused.size())
        return fused



    def encoder(self, x, x_c, input_id, d_emb):
        c1, inv_x1 = self.conv1(x, x_c, input_id, d_emb)
        p1 = self.pool1(c1)
        inv_x1 = self.pool1(inv_x1)
        c2, inv_x2 = self.conv2(p1, inv_x1, input_id, d_emb)
        p2 = self.pool2(c2)
        inv_x2 = self.pool1(inv_x2)
        c3, inv_x3 = self.conv3(p2, inv_x2, input_id, d_emb)
        p3 = self.pool3(c3)
        inv_x3 = self.pool1(inv_x3)
        c4, inv_x4 = self.conv4(p3, inv_x3, input_id, d_emb)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4, d_emb)
        return c1, c2, c3, c4, c5

if __name__ == "__main__":

    # unet
    unet = Generator(1, 1, 6)
    print(unet)
    for name, p in unet.named_parameters():
        if p.requires_grad==True:
            print(name)