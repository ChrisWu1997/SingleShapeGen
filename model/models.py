import torch
from .model_utils import generate_tri_plane_noise, generate_3d_noise, make_coord
from .model_base import SSGmodelBase


class SSGmodelTriplane(SSGmodelBase):
    def _netG_trainable_params(self, lr_g, lr_sigma, train_depth):
        # set different learning rate for lower stages
        parameter_list = [{"params": block.parameters(), "lr": lr_g * (lr_sigma ** (len(self.netG.body[-train_depth:]) - 1 - idx))}
                for idx, block in enumerate(self.netG.body[-train_depth:])]
        # add parameters of head and tail to training
        depth = self.netG.n_scales - 1
        if depth - train_depth < 0:
            parameter_list += [{"params": self.netG.head_conv.parameters(), "lr": lr_g * (lr_sigma ** depth)}]
        parameter_list += [{"params": self.netG.mlp.parameters(), "lr": lr_g}]
        return parameter_list
    
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
        noises_list = self.draw_noises_list(mode, scale, resize_factor)

        coords = None
        if upsample > 1:
            query_shape = [round(x * upsample) for x in real_sizes[-1]]
            coords = make_coord(*query_shape, self.device)
        out = self.netG(init_noise, real_sizes, noises_list, mode, coords, return_each=return_each)
        return out

    def interpolation(self, alpha_list):
        mode = 'rand'
        init_noise1 = self.draw_init_noise(mode)
        init_noise2 = self.draw_init_noise(mode)
        noises_list = self.draw_noises_list(mode)

        out_list = []
        with torch.no_grad():
            for i in range(len(alpha_list)):
                alpha = alpha_list[i]
                init_noise = (1 - alpha) * init_noise1 + alpha * init_noise2
                out = self.netG(init_noise, self.real_sizes, noises_list, mode)
                out_list.append(out)
        return out_list


class SSGmodelConv3D(SSGmodelBase):
    def _netG_trainable_params(self, lr_g, lr_sigma, train_depth):
        # set different learning rate for lower stages
        parameter_list = [{"params": block.parameters(), "lr": lr_g * (lr_sigma ** (len(self.netG.body[-train_depth:]) - 1 - idx))}
                for idx, block in enumerate(self.netG.body[-train_depth:])]
        return parameter_list
    
    def _draw_fake_in_training(self, mode):
        init_inp = torch.zeros_like(self.noiseOpt_init)
        noises_list = self.draw_noises_list(mode, self.scale)

        if self.scale < self.train_depth:
            fake = self.netG(init_inp, noises_list)
        else:
            prev_depth = self.scale - self.train_depth
            if mode == 'rec' and self.prev_opt_feats is not None:
                prev_feats = self.prev_opt_feats
            else:
                with torch.no_grad():
                    prev_feats = self.netG(init_inp, noises_list[:prev_depth + 1], end_scale=prev_depth)
                prev_feats = prev_feats.detach()
                if mode == 'rec' and self.prev_opt_feats is None:
                    self.prev_opt_feats = prev_feats
            fake = self.netG(prev_feats, noises_list[prev_depth + 1:], start_scale=prev_depth + 1)
        return fake

    def draw_noises_list(self, mode, scale, resize_factor=(1.0, 1.0, 1.0)):
        noises_list = [] # first scale no additive noise
        for i in range(scale + 1):
            if i == 0:
                noise = self.draw_init_noise(mode, resize_factor)
            else:
                noise_shape = self.real_sizes[i]
                if resize_factor != (1.0, 1.0, 1.0):
                    noise_shape = [round(noise_shape[j] * resize_factor[j]) for j in range(3)]
                noise = generate_3d_noise(*noise_shape, mode, self.noiseAmp_list[i], self.device)
            noises_list.append(noise)
        return noises_list
    
    def generate(self, mode, scale=None, resize_factor=(1.0, 1.0, 1.0), upsample=1, return_each=False):
        if scale is None:
            scale = self.scale
        noises_list = self.draw_noises_list(mode, scale, resize_factor)
        init_inp = torch.zeros_like(noises_list[0])
        # if resize_factor != (1.0, 1.0, 1.0):
        #     assert mode != 'rec'
        #     init_size = [round(self.noiseOpt_init.shape[-3:][i] * resize_factor[i]) for i in range(3)]
        #     init_inp = torch.zeros(init_size, dtype=torch.float32).cuda()
        #     # init_noise = torch.randn_like(init_inp)
        #     # noises_list[0] = init_noise
        # else:
        #     init_inp = self.template
        
        out = self.netG(init_inp, noises_list, end_scale=scale, return_each=return_each)
        return out
