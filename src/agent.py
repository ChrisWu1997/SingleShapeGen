import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
from tensorboardX import SummaryWriter
import numpy as np
from utils import TrainClock
from networks import get_network
from helpers import draw_mat_figure_along_xyz


def get_agent(config):
    return SinGANAgent(config)


class SinGANAgent(object):
    def __init__(self, config):
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.clock = TrainClock()
        self.config = config
        self.train_depth = config.train_depth

        self.scale = 0
        self.netD = get_network(config, 'D').cuda()
        self.netG = get_network(config, 'G').cuda()
        # self.noiseOpt_list = []
        self.noiseOpt_init = None # assume rec use all zero noise for scale > 0
        self.noiseAmp_list = []

        self.device = torch.device('cuda:0')

    def set_optimizer(self, config):
        """set optimizer and lr scheduler used in training"""
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=config.lr_d, betas=(config.beta1, 0.999))
        # self.optimizerG = optim.Adam(self.netG.trainable_parameters(), lr=config.lr_g, betas=(config.beta1, 0.999))
        # set different learning rate for lower stages
        parameter_list = [{"params": block.parameters(), "lr": config.lr_g * (config.sigma ** (len(self.netG.body[-config.train_depth:]) - 1 - idx))}
                for idx, block in enumerate(self.netG.body[-config.train_depth:])]

        # add parameters of head and tail to training
        depth = self.netG.n_stage - 1
        if depth - config.train_depth < 0:
            parameter_list += [{"params": self.netG.head_conv.parameters(), "lr": config.lr_g * (config.sigma ** depth)}]
        parameter_list += [{"params": self.netG.mlp.parameters(), "lr": config.lr_g}]
        print([x['lr'] for x in parameter_list])
        self.optimizerG = optim.Adam(parameter_list, lr=config.lr_g, betas=(config.beta1, 0.999))

    def set_scheduler(self, config):
        self.schedulerD = None
        self.schedulerG = None
    
    def set_tbwriter(self):
        path = os.path.join(self.log_dir, 'train_s{}.events'.format(self.scale))
        self.train_tb = SummaryWriter(path)

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_scale{}_step{}.pth".format(self.scale, self.clock.step))
            # print("Saving checkpoint step {}...".format(self.clock.step))
        else:
            save_path = os.path.join(self.model_dir, "scale{}_{}.pth".format(self.scale, name))

        # noise_opt_list = [self.noiseOpt_list[-1].detach().cpu()] if self.scale == 0 else [x.detach().cpu() for x in self.noiseOpt_list[-1]]
        torch.save({
            'clock': self.clock.make_checkpoint(),
            'netD_state_dict': self.netD.cpu().state_dict(),
            'netG_state_dict': self.netG.cpu().state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'schedulerD_state_dict': self.schedulerD.state_dict() if self.schedulerD is not None else None,
            'schedulerG_state_dict': self.schedulerG.state_dict() if self.schedulerG is not None else None,
            'noiseOpt_init': self.noiseOpt_init.detach().cpu(),
            'noiseAmp_list': self.noiseAmp_list,
        }, save_path)

        self.netD.cuda()
        self.netG.cuda()

    def load_ckpt(self, n_scale):
        """load checkpoint from saved checkpoint"""
        s = n_scale - 1
        load_path = os.path.join(self.model_dir, "scale{}_latest.pth".format(s))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))
        print("Loading checkpoint from {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        
        self.noiseOpt_init = checkpoint['noiseOpt_init'].cuda()
        self.noiseAmp_list = checkpoint['noiseAmp_list']
        for s in range(n_scale):
            self.netG.init_next_stage()
        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.netD.load_state_dict(checkpoint['netD_state_dict'])
        self.netG.cuda()
        self.netD.cuda()

        self.set_optimizer(self.config)
        self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.set_scheduler(self.config)
        if self.schedulerD is not None:
            self.schedulerD.load_state_dict(checkpoint['schedulerD_state_dict'])
        if self.schedulerG is not None:
            self.schedulerG.load_state_dict(checkpoint['schedulerG_state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])

        self.scale = n_scale

    def update_learning_rate(self):
        """record and update learning rate"""
        if self.schedulerD is not None:
            self.train_tb.add_scalar('lrD', self.optimizerD.param_groups[-1]['lr'], self.clock.step)
            self.schedulerD.step()
        if self.schedulerG is not None:
            self.train_tb.add_scalar('lrG', self.optimizerG.param_groups[-1]['lr'], self.clock.step)
            self.schedulerG.step()

    def _calc_gradient_penalty(self, netD, real_data, fake_data):
        #print real_data.size()
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda() #gpu) #if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).cuda(), #if use_cuda else torch.ones(
                                    #disc_interpolates.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        #LAMBDA = 1
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        # gradient_penalty = torch.clamp((gradients.norm(2, dim=1) - 1) ** 2, min=None, max=1.0).mean()
        return gradient_penalty

    def _set_require_grads(self, model, require_grad):
        for p in model.parameters():
            p.requires_grad_(require_grad)

    def _draw_fake_in_training(self, mode):
        init_inp = self.template
        init_noise = self.draw_init_noise(mode)
        real_shapes = self.real_shapes[:self.scale + 1]
        noises_list = self.draw_noises_list(mode, self.scale)

        if self.scale < self.train_depth:
            fake = self.netG(init_noise, init_inp, real_shapes, noises_list, mode)
        else:
            prev_depth = self.scale - self.train_depth
            if mode == 'rec' and self.prev_opt_feats is not None:
                prev_feats = self.prev_opt_feats
            else:
                with torch.no_grad():
                    prev_feats = self.netG.draw_feats(init_noise, init_inp, 
                        real_shapes[:prev_depth + 1], noises_list[:prev_depth + 1], mode, prev_depth + 1)
                prev_feats = [x.detach() for x in prev_feats]
                if mode == 'rec' and self.prev_opt_feats is None:
                    self.prev_opt_feats = prev_feats
            fake = self.netG.decode_feats(prev_feats, real_shapes[prev_depth + 1:], noises_list[prev_depth + 1:], 
                    mode, prev_depth + 1, -1)
        return fake

    def _critic_wgan_iteration(self, real_data):
        # require grads
        self._set_require_grads(self.netD, True)

        # get generated data
        # generated_data = self.generate('rand', self.scale)
        generated_data = self._draw_fake_in_training('rand')

        # zero grads
        self.optimizerD.zero_grad()

        # calculate probabilities on real and generated data
        d_real = self.netD(real_data)
        d_generated = self.netD(generated_data.detach())

        # create total loss and optimize
        loss_r = -d_real.mean()
        loss_f = d_generated.mean()
        loss = loss_f + loss_r

        # get gradient penalty
        if self.config.lambda_grad:
            gradient_penalty = self._calc_gradient_penalty(self.netD, real_data, generated_data) * self.config.lambda_grad
            loss += gradient_penalty 
        
        # backward loss
        loss.backward()
        self.optimizerD.step()

        # record loss
        self.losses['D'].append(loss.data.item())
        self.losses['D_r'].append(loss_r.data.item())
        self.losses['D_f'].append(loss_f.data.item())
        if self.config.lambda_grad:
            self.losses['D_gp'].append(gradient_penalty.data.item())

    def _generator_iteration(self, real_data):
        # require grads
        self._set_require_grads(self.netD, False)

        # zero grads
        self.optimizerG.zero_grad()

        loss = 0.

        # adversarial loss
        # fake_data = self.generate('rand', self.scale)
        fake_data = self._draw_fake_in_training('rand')

        d_generated = self.netD(fake_data)
        loss_adv = -d_generated.mean()
        loss += loss_adv

        # reconstruction loss
        if self.config.alpha:
            # generated_data_rec = self.generate('rec', self.scale)
            generated_data_rec = self._draw_fake_in_training('rec')
            loss_recon = F.mse_loss(generated_data_rec, real_data) * self.config.alpha
            loss += loss_recon
        
        # backward loss
        loss.backward()
        self.optimizerG.step()

        # record loss
        self.losses['G'].append(loss.data.item())
        self.losses['G_adv'].append(loss_adv.data.item())
        if self.config.alpha:
            self.losses['G_rec'].append(loss_recon.data.item())

    def set_losses(self):
        if self.config.alpha > 0:
            self.losses = {'D': [], 'D_r': [], 'D_gp': [], 'D_f': [], 'G': [], 'G_rec': [], 'G_adv': []}
        else:
            self.losses = {'D': [], 'D_r': [], 'D_gp': [], 'D_f': [], 'G': [], 'G_adv': []}

    def record_losses(self):
        self.train_tb.add_scalars("loss", {k: np.mean(v) for k, v in self.losses.items()}, global_step=self.clock.step)
        Wasserstein_D = np.mean([-self.losses['D_r'][i] - self.losses['D_f'][i] for i in range(len(self.losses['D_r']))])
        self.train_tb.add_scalar("wasserstein distance", Wasserstein_D, global_step=self.clock.step)
    
    def set_real_data(self, real_data_list):
        print("real data resolutions: ", [x.shape for x in real_data_list])
        self.template = torch.zeros_like(real_data_list[0]) if not self.config.use_temp else real_data_list[0]
        self.real_list = real_data_list[1:]
        self.real_shapes = [x.shape[-3:] for x in self.real_list]

    def train(self, real_data_list):
        self.set_real_data(real_data_list)
        self.n_scales = len(self.real_list)
        
        for s in range(self.scale, self.n_scales):

            self.netG.init_next_stage()
            self.netG.cuda()
            assert self.netG.n_stage == s + 1
            # self.netD is reused
            # self.netD = get_network(self.config, 'D')

            self.set_optimizer(self.config)
            self.set_scheduler(self.config)
            self.set_tbwriter()
            self.clock.reset()
            
            if self.noiseOpt_init is None:
                torch.manual_seed(1234)
                self.noiseOpt_init = torch.randn_like(self.template)

            # init networks and optimizers for each scale
            if self.config.alpha > 0:
                if s > 0:
                    # prev_rec = self.netG(self.noiseOpt_init, self.template, self.real_shapes[:s], self.noiseAmp_list[:s], 'rec')
                    prev_rec = self.generate('rec', s - 1)
                    prev_rec = F.interpolate(prev_rec, size=self.real_list[s].shape[2:], mode='trilinear', align_corners=False)
                    noise_amp = self.config.init_noise_amp * torch.sqrt(F.mse_loss(self.real_list[s], prev_rec))
                else:
                    noise_amp = 1.0
            else:
                noise_amp = 0.1 if s > 0 else self.config.init_noise_amp
            self.noiseAmp_list.append(noise_amp)

            # train for current scale
            self.train_single_scale(self.real_list[s])

            self.scale += 1

    def train_single_scale(self, real_data):
        print("scale: {}, real shape: {}, noise amp: {}".format(self.scale, real_data.shape, self.noiseAmp_list[-1]))
        pbar = tqdm(range(self.config.n_iters))
        self.prev_opt_feats = None
        for i in pbar:
            losses = self.updateStep(real_data)
            pbar.set_description("EPOCH[{}][{}]".format(i, self.config.n_iters))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            self.update_learning_rate()

            if self.config.vis_frequency is not None and self.clock.step % self.config.vis_frequency == 0:
                self.visualize_batch(real_data)

            self.clock.tick()

            if self.clock.step % self.config.save_frequency == 0:
                self.save_ckpt()
        
        self.prev_opt_feats = None
        self.save_ckpt('latest')

    def updateStep(self, real_data):
        self.set_losses()
        self.netD.train()
        self.netG.train()

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(self.config.Dsteps):
            self._critic_wgan_iteration(real_data)
        
        ############################
        # (2) Update G network: maximize D(G(z)) + rec(G(noise_opt), real_data)
        ###########################
        for j in range(self.config.Gsteps):
            self._generator_iteration(real_data)

        self.record_losses()

        if self.config.alpha > 0:
            return {'D': np.mean(self.losses['D']), 'G_adv': np.mean(self.losses['G_adv']), 'G_rec': np.mean(self.losses['G_rec'])}
        else:
            return {'D': np.mean(self.losses['D']), 'G_adv': np.mean(self.losses['G_adv'])}

    def draw_init_noise(self, mode):
        if mode == 'rec':
            return self.noiseOpt_init
        else:
            return torch.randn_like(self.noiseOpt_init)
    
    def _generate_tri_plane_noise(self, res_x, res_y, res_z, nf, noise_amp, device):
        noise = [(torch.randn(1, nf, res_y, res_z, device=device) * noise_amp).detach(), 
                 (torch.randn(1, nf, res_x, res_z, device=device) * noise_amp).detach(), 
                 (torch.randn(1, nf, res_x, res_y, device=device) * noise_amp).detach()]
        return noise

    def draw_noises_list(self, mode, scale, resize_factor=(1.0, 1.0, 1.0)):
        noises_list = [] # first scale no additive noise
        for i in range(scale + 1):
            if i == 0:
                noises_list.append(None)
            else:
                if mode == 'rec':
                    noises_list.append([0, 0, 0])
                else:
                    noise_shape = self.real_shapes[i]
                    if resize_factor != (1.0, 1.0, 1.0):
                        noise_shape = [round(noise_shape[j] * resize_factor[j]) for j in range(3)]
                    tri_noise = self._generate_tri_plane_noise(*noise_shape, self.config.feat_dim, self.noiseAmp_list[i], self.device)
                    noises_list.append(tri_noise)
        return noises_list
    
    def generate(self, mode, scale, resize_factor=(1.0, 1.0, 1.0), return_each=False):
        if resize_factor != (1.0, 1.0, 1.0):
            assert mode != 'rec'
            init_size = [round(self.template.shape[-3:][i] * resize_factor[i]) for i in range(3)]
            init_inp = F.interpolate(self.template, size=init_size, mode='trilinear', align_corners=True)
            init_noise = torch.randn_like(init_inp)
            real_shapes = [[round(x[i] * resize_factor[i]) for i in range(3)] for x in self.real_shapes[:scale + 1]]
        else:
            init_inp = self.template
            init_noise = self.draw_init_noise(mode)
            real_shapes = self.real_shapes[:scale + 1]
        
        noises_list = self.draw_noises_list(mode, scale, resize_factor)
        out = self.netG(init_noise, init_inp, real_shapes, noises_list, mode, return_each=return_each)
        return out

    def visualize_batch(self, real_data):
        if self.clock.step == 0:
            real_data_ = real_data.detach().cpu().numpy()[0, 0]
            self.train_tb.add_figure('real', draw_mat_figure_along_xyz(real_data_), self.clock.step)

        with torch.no_grad():
            fake1_ = self.generate('rand', self.scale)
            rec_ = self.generate('rec', self.scale)

        fake1_ = fake1_.detach().cpu().numpy()[0, 0]
        self.train_tb.add_figure('fake1', draw_mat_figure_along_xyz(fake1_), self.clock.step)
        self.train_tb.add_figure('fake1_max', draw_mat_figure_along_xyz(fake1_, mid=False, grayscale=False), self.clock.step)

        rec_ = rec_.detach().cpu().numpy()[0, 0]
        self.train_tb.add_figure('rec', draw_mat_figure_along_xyz(rec_), self.clock.step)
        self.train_tb.add_figure('rec_max', draw_mat_figure_along_xyz(rec_, mid=False, grayscale=False), self.clock.step)
