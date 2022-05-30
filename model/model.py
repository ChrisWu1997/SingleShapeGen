import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
from tensorboardX import SummaryWriter
import numpy as np
from .networks import get_network
from .model_utils import calc_gradient_penalty, set_require_grads, generate_tri_plane_noise, draw_mat_figure_along_xyz, TrainClock


class SSGmodel(object):
    def __init__(self, config):
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.clock = TrainClock()
        self.config = config
        self.train_depth = config.train_depth

        self.scale = 0
        self.netD = get_network(config, 'D').cuda()
        self.netG = get_network(config, 'G').cuda()
        self.noiseOpt_init = None # assume rec use all zero noise for scale > 0
        self.noiseAmp_list = [] # gaussian noise std for each scale
        self.real_sizes = [] # real data spatial dimensions

        self.device = torch.device('cuda:0')

    def _set_optimizer(self, config):
        """set optimizer used in training"""
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=config.lr_d, betas=(config.beta1, 0.999))
    
        # set different learning rate for lower stages
        parameter_list = [{"params": block.parameters(), "lr": config.lr_g * (config.lr_sigma ** (len(self.netG.body[-config.train_depth:]) - 1 - idx))}
                for idx, block in enumerate(self.netG.body[-config.train_depth:])]

        # add parameters of head and tail to training
        depth = self.netG.n_scales - 1
        if depth - config.train_depth < 0:
            parameter_list += [{"params": self.netG.head_conv.parameters(), "lr": config.lr_g * (config.lr_sigma ** depth)}]
        parameter_list += [{"params": self.netG.mlp.parameters(), "lr": config.lr_g}]
        # print([x['lr'] for x in parameter_list])
        self.optimizerG = optim.Adam(parameter_list, lr=config.lr_g, betas=(config.beta1, 0.999))
    
    def _set_tbwriter(self):
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
            'noiseOpt_init': self.noiseOpt_init.detach().cpu(),
            'noiseAmp_list': self.noiseAmp_list,
            'realSizes_list': self.real_sizes,
        }, save_path)

        self.netD.cuda()
        self.netG.cuda()

    def load_ckpt(self, n_scale):
        """load checkpoint from saved checkpoint"""
        load_path = os.path.join(self.model_dir, "scale{}_latest.pth".format(n_scale))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))
        print("Loading checkpoint from {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        
        self.noiseOpt_init = checkpoint['noiseOpt_init'].cuda()
        self.noiseAmp_list = checkpoint['noiseAmp_list']
        self.real_sizes = checkpoint['realSizes_list']
        for _ in range(n_scale + 1):
            self.netG.init_next_scale()
        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.netD.load_state_dict(checkpoint['netD_state_dict'])
        self.netG.cuda()
        self.netD.cuda()

        self._set_optimizer(self.config)
        self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])

        self.scale = n_scale

    def _draw_fake_in_training(self, mode):
        init_noise = self.draw_init_noise(mode)
        real_sizes = self.real_sizes[:self.scale + 1]
        noises_list = self.draw_noises_list(mode, self.scale)

        if self.scale < self.train_depth:
            fake = self.netG(init_noise, real_sizes, noises_list, mode)
        else:
            # NOTE: get features from non-trainable scales under torch.no_grad(), seems to be quicker
            prev_depth = self.scale - self.train_depth
            if mode == 'rec' and self.prev_opt_feats is not None:
                prev_feats = self.prev_opt_feats
            else:
                with torch.no_grad():
                    prev_feats = self.netG.draw_feats(init_noise, 
                        real_sizes[:prev_depth + 1], noises_list[:prev_depth + 1], mode, prev_depth + 1)
                prev_feats = [x.detach() for x in prev_feats]
                if mode == 'rec' and self.prev_opt_feats is None:
                    self.prev_opt_feats = prev_feats
            fake = self.netG.decode_feats(prev_feats, real_sizes[prev_depth + 1:], noises_list[prev_depth + 1:], 
                    mode, prev_depth + 1, -1)
        return fake

    def _critic_wgan_iteration(self, real_data):
        # require grads
        set_require_grads(self.netD, True)

        # get generated data
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
            gradient_penalty = calc_gradient_penalty(self.netD, real_data, generated_data) * self.config.lambda_grad
            loss += gradient_penalty 
        
        # backward loss
        loss.backward()
        self.optimizerD.step()

        # record loss
        loss_values = {'D': loss.data.item(), 'D_r': loss_r.data.item(), 'D_f': loss_f.data.item()}
        if self.config.lambda_grad:
            loss_values.update({'D_gp': gradient_penalty.data.item()})
        self._update_loss_dict(loss_values)

    def _generator_iteration(self, real_data):
        # require grads
        set_require_grads(self.netD, False)

        # zero grads
        self.optimizerG.zero_grad()
        loss = 0.

        # adversarial loss
        fake_data = self._draw_fake_in_training('rand')

        d_generated = self.netD(fake_data)
        loss_adv = -d_generated.mean()
        loss += loss_adv

        # reconstruction loss
        if self.config.alpha:
            generated_data_rec = self._draw_fake_in_training('rec')
            loss_recon = F.mse_loss(generated_data_rec, real_data) * self.config.alpha
            loss += loss_recon
        
        # backward loss
        loss.backward()
        self.optimizerG.step()

        # record loss
        loss_values = {'G': loss.data.item(), 'G_adv': loss_adv.data.item()}
        if self.config.alpha:
            loss_values.update({'G_rec': loss_recon.data.item()})
        self._update_loss_dict(loss_values)
    
    def _update_loss_dict(self, loss_dict: dict=None):
        if loss_dict is None:
            self.losses = {}
        else:
            for k, v in loss_dict.items():
                if k in self.losses:
                    self.losses[k].append(v)
                else:
                    self.losses[k] = [v]

    def _record_losses(self):
        avg_loss = {k: np.mean(v) for k, v in self.losses.items()}
        self.train_tb.add_scalars("loss", avg_loss, global_step=self.clock.step)
        Wasserstein_D = np.mean([-self.losses['D_r'][i] - self.losses['D_f'][i] for i in range(len(self.losses['D_r']))])
        self.train_tb.add_scalar("wasserstein distance", Wasserstein_D, global_step=self.clock.step)
        return avg_loss
    
    def _updateStep(self, real_data):
        self._update_loss_dict(None)
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

        avg_loss = self._record_losses()
        if self.config.alpha > 0:
            return {'D': avg_loss['D'], 'G_adv': avg_loss['G_adv'], 'G_rec': avg_loss['G_rec']}
        return {'D': avg_loss['D'], 'G_adv': avg_loss['G_adv']}

    def _train_single_scale(self, real_data):
        print("scale: {}, real shape: {}, noise amp: {}".format(self.scale, real_data.shape, self.noiseAmp_list[-1]))
        pbar = tqdm(range(self.config.n_iters))
        self.prev_opt_feats = None # buffer of prev scale features for reconstruction
        for i in pbar:
            losses = self._updateStep(real_data)
            pbar.set_description("EPOCH[{}][{}]".format(i, self.config.n_iters))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            if self.config.vis_frequency is not None and self.clock.step % self.config.vis_frequency == 0:
                self._visualize_in_training(real_data)

            self.clock.tick()

            if self.clock.step % self.config.save_frequency == 0:
                self.save_ckpt()
        
        self.prev_opt_feats = None
        self.save_ckpt('latest')

    def train(self, real_data_list):
        self._set_real_data(real_data_list)
        self.n_scales = len(self.real_list)
        
        for s in range(self.scale, self.n_scales):
            # init networks and optimizers for each scale
            # self.netD is reused directly
            self.netG.init_next_scale()
            self.netG.cuda()
            assert self.netG.n_scales == s + 1

            self._set_optimizer(self.config)
            self._set_tbwriter()
            self.clock.reset()
            
            # draw fixed noise for reconstruction
            if self.noiseOpt_init is None:
                torch.manual_seed(1234)
                self.noiseOpt_init = torch.randn_like(self.real_list[0])

            # draw gaussian noise std
            noise_amp = self._compute_noise_sigma(s)
            self.noiseAmp_list.append(noise_amp)

            # train for current scale
            self._train_single_scale(self.real_list[s])

            self.scale += 1

    def _set_real_data(self, real_data_list):
        print("real data resolutions: ", [x.shape for x in real_data_list])
        self.real_list = [torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda() for x in real_data_list]
        self.real_sizes = [x.shape[-3:] for x in self.real_list]

    def _compute_noise_sigma(self, scale):
        s = scale
        if self.config.alpha > 0:
            if s > 0:
                prev_rec = self.generate('rec', s - 1)
                prev_rec = F.interpolate(prev_rec, size=self.real_list[s].shape[2:], mode='trilinear', align_corners=False)
                noise_amp = self.config.base_noise_amp * torch.sqrt(F.mse_loss(self.real_list[s], prev_rec))
            else:
                noise_amp = 1.0
        else:
            noise_amp = self.config.base_noise_amp if s > 0 else 1.0
        return noise_amp

    def draw_init_noise(self, mode, resize_factor=(1.0, 1.0, 1.0)):
        if mode == 'rec':
            return self.noiseOpt_init
        else:
            if resize_factor[0] != 1.0 or resize_factor[1] != 1.0 or resize_factor[2] != 1.0:
                init_size = [round(self.real_sizes[i] * resize_factor[i]) for i in range(3)]
                return torch.randn(*init_size, device=self.device)
            return torch.randn_like(self.noiseOpt_init)
    
    def draw_noises_list(self, mode, scale=None, resize_factor=(1.0, 1.0, 1.0)):
        if scale is None:
            scale = self.scale
        noises_list = [] # first scale no additive noise
        for i in range(scale + 1):
            if i == 0:
                noises_list.append(None)
            else:
                if mode == 'rec':
                    noises_list.append([0, 0, 0])
                else:
                    noise_shape = self.real_sizes[i]
                    if resize_factor[0] != 1.0 or resize_factor[1] != 1.0 or resize_factor[2] != 1.0:
                        noise_shape = [round(noise_shape[j] * resize_factor[j]) for j in range(3)]
                    tri_noise = generate_tri_plane_noise(*noise_shape, self.config.feat_dim, self.noiseAmp_list[i], self.device)
                    noises_list.append(tri_noise)
        return noises_list
    
    def generate(self, mode, scale=None, resize_factor=(1.0, 1.0, 1.0), upsample=1, return_each=False):
        if scale is None:
            scale = self.scale
        init_noise = self.draw_init_noise(mode, resize_factor)
        real_sizes = [[round(x[i] * resize_factor[i]) for i in range(3)] for x in self.real_sizes[:scale + 1]]
        query_shape = [round(x * upsample) for x in real_sizes[-1]]
        
        noises_list = self.draw_noises_list(mode, scale, resize_factor)
        out = self.netG(init_noise, real_sizes, noises_list, mode, return_each=return_each)
        return out

    def _visualize_in_training(self, real_data):
        if self.clock.step == 0:
            real_data_ = real_data.detach().cpu().numpy()[0, 0]
            self.train_tb.add_figure('real', draw_mat_figure_along_xyz(real_data_), self.clock.step)

        with torch.no_grad():
            fake1_ = self.generate('rand', self.scale)
            rec_ = self.generate('rec', self.scale)

        fake1_ = fake1_.detach().cpu().numpy()[0, 0]
        self.train_tb.add_figure('fake1', draw_mat_figure_along_xyz(fake1_), self.clock.step)
        rec_ = rec_.detach().cpu().numpy()[0, 0]
        self.train_tb.add_figure('rec', draw_mat_figure_along_xyz(rec_), self.clock.step)
