import torch
import torch.nn as nn
import torch.nn.functional as F
# from helpers import nonempty_surf_coords_more, nonempty_coords


def get_network(config, name):
    if name == 'G':
        return GrowingGeneratorTriplane(config.feat_dim, config.use_norm, pad_head=False)
    elif name == 'D':
        return WDiscriminator(config)
    else:
        raise NotImplementedError


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride, use_norm=True, sdim='3d', padd_mode='zeros', onlyconv=False):
        super(ConvBlock, self).__init__()
        if sdim == '3d':
            self.add_module('conv',nn.Conv3d(in_channel, out_channel, kernel_size=ker_size, stride=stride, 
                padding=padd, padding_mode=padd_mode, bias=use_norm is False))
        else:
            self.add_module('conv',nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, 
                padding=padd, padding_mode=padd_mode, bias=use_norm is False))
        if not onlyconv:
            if use_norm:
                if sdim == '3d':
                    # self.add_module('norm',nn.BatchNorm3d(out_channel))
                    self.add_module('norm',nn.InstanceNorm3d(out_channel))
                else:
                    # self.add_module('norm',nn.BatchNorm2d(out_channel))
                    self.add_module('norm',nn.InstanceNorm2d(out_channel))
            self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv3d') != -1 or classname.find('Conv2d') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('Norm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        # self.is_cuda = torch.cuda.is_available()
        N = nfc = 32 # int(opt.nfc)
        min_nfc = 32
        # self.head = ConvBlock(1, N, opt.ker_size, 0, 1, opt.use_norm)
        self.head = ConvBlock(1, N, opt.ker_size, 1, 2, opt.use_norm)
        self.body = nn.Sequential()
        for i in range(1):
            N = int(nfc / pow(2, (i + 1)))
            stride = 1
            pad = 0
            block = ConvBlock(max(2 * N, min_nfc),max(N, min_nfc), opt.ker_size, pad, stride, opt.use_norm)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv3d(max(N, min_nfc), 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


def make_mlp(in_features, out_features, hidden_features, n_hidden_layers, act=nn.ReLU(inplace=True)):
    layer_list = [nn.Linear(in_features, hidden_features), act]
    for i in range(n_hidden_layers):
        layer_list.extend([nn.Linear(hidden_features, hidden_features), act])
    layer_list.extend([nn.Linear(hidden_features, out_features)])
    return nn.Sequential(*layer_list)


class TriplaneConvs(nn.Module):
    def __init__(self, in_dim, hidden_list, use_norm=True, pad=0):
        super(TriplaneConvs, self).__init__()
        if type(hidden_list) is int:
            out_dim = hidden_list
            self.conv_yz = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
            self.conv_xz = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
            self.conv_xy = nn.Conv2d(in_dim, out_dim, 1, 1, 0)
        else:
            self.conv_yz = self.make_2Dconvs(in_dim, hidden_list, pad, use_norm)
            self.conv_xz = self.make_2Dconvs(in_dim, hidden_list, pad, use_norm)
            self.conv_xy = self.make_2Dconvs(in_dim, hidden_list, pad, use_norm)

    def make_2Dconvs(self, in_dim, hidden_list, pad, use_norm):
        model = nn.Sequential()
        model.add_module('conv0', ConvBlock(in_dim, hidden_list[0], 3, pad, 1, 
            use_norm, sdim='2d', padd_mode='zeros'))

        for i in range(len(hidden_list) - 1):
            is_last = i == len(hidden_list) - 2
            use_norm = use_norm if not is_last else False
            model.add_module(f'conv{i + 1}', ConvBlock(hidden_list[i], hidden_list[i + 1], 3, pad, 1, 
                use_norm, sdim='2d', padd_mode='zeros', onlyconv=is_last))
        return model

    def forward(self, tri_feats):
        yz_feat = self.conv_yz(tri_feats[0])
        xz_feat = self.conv_xz(tri_feats[1])
        xy_feat = self.conv_xy(tri_feats[2])
        return yz_feat, xz_feat, xy_feat


class GrowingGeneratorTriplane(nn.Module):
    def __init__(self, feat_dim=32, use_norm=True, mlp_dim=32, mlp_layers=0, pad_head=False):
        super(GrowingGeneratorTriplane, self).__init__()
        # self.is_cuda = torch.cuda.is_available()
        self.avg_dim = 8
        self.nf = 32
        self.n_conv_layers = 4
        self.feat_dim = feat_dim
        self.use_norm = use_norm
        self.pad_len = 0 if pad_head else 1
        
        self.head_conv = TriplaneConvs(self.avg_dim, self.feat_dim, use_norm=use_norm, pad=0)
        
        self.body = nn.ModuleList([])
        
        # NOTE: concat tri-plane features, position invariant
        self.mlp = make_mlp(self.feat_dim * 3, 1, mlp_dim, mlp_layers)
        # self.mlp.add_module('sigmoid', nn.Sigmoid()) # restrict range to 0 ~ 1
        
        self.pad_head = pad_head
        if pad_head:
            self.pad_block = nn.ZeroPad2d(self.n_conv_layers)

        # self.noise_amp_list = nn.ParameterList()

    @property
    def n_stage(self):
        return len(self.body)

    def init_next_stage(self):
        model = TriplaneConvs(self.feat_dim, [self.nf] * (self.n_conv_layers - 1) + [self.feat_dim], 
            use_norm=self.use_norm, pad=self.pad_len)
        # if len(self.body) > 0:
        #     print("load generator body weights from previous level.")
        #     model.load_state_dict(self.body[-1].state_dict())
        self.body.append(model)
    
    def query(self, tri_feats, coords=None):
        """_summary_

        Args:
            tri_feats (_type_): [yz_feat, xz_feat, xy_feat]
            coords (_type_, optional): (..., 3)
        """
        yz_feat, xz_feat, xy_feat = tri_feats
        in_shape = [*xy_feat.shape[-2:], yz_feat.shape[-1]]
    
        if coords is None:
            yz_feat = yz_feat.permute(0, 2, 3, 1) # (1, W, D, nf)
            xz_feat = xz_feat.permute(0, 2, 3, 1) # (1, H, D, nf)
            xy_feat = xy_feat.permute(0, 2, 3, 1) # (1, H, W, nf)

            vol_feat = torch.cat([yz_feat.unsqueeze(1).expand(1, *in_shape, -1),
                                xz_feat.unsqueeze(2).expand(1, *in_shape, -1),
                                xy_feat.unsqueeze(3).expand(1, *in_shape, -1)], dim=-1)

            out = self.mlp(vol_feat).permute(0, 4, 1, 2, 3)
        else:
            # coords_ = coords.view(-1, 3)
            # yz_feat = yz_feat[:, coords_[:, 1], coords_[:, 2], :] # (1, N, nf)
            # xz_feat = xz_feat[:, coords_[:, 0], coords_[:, 2], :] # (1, N, nf)
            # xy_feat = xy_feat[:, coords_[:, 0], coords_[:, 1], :] # (1, N, nf)
            # vol_feat = torch.cat([yz_feat, xz_feat, xy_feat], dim=-1).squeeze(0) # (N, nf)
                        # out = self.mlp(vol_feat).squeeze(-1).view(*coords.shape[:-1])

            # coords shape: (H, W, D, 3)
            c_shape = coords.shape[:3]
            coords = coords.view(-1, 3)
            # coords = coords.view(-1, 3).unsqueeze(0).unsqueeze(0) # (1, 1, N, 3)
            # sample_yz_feat = F.grid_sample(yz_feat, coords[..., [1, 2]].flip(-1), align_corners=True).permute(0, 2, 3, 1).squeeze(0).squeeze(0)
            # sample_xz_feat = F.grid_sample(xz_feat, coords[..., [0, 2]].flip(-1), align_corners=True).permute(0, 2, 3, 1).squeeze(0).squeeze(0)
            # sample_xy_feat = F.grid_sample(xy_feat, coords[..., [0, 1]].flip(-1), align_corners=True).permute(0, 2, 3, 1).squeeze(0).squeeze(0)
            
            # vol_feat = torch.cat([sample_yz_feat, sample_xz_feat, sample_xy_feat], dim=-1) # (N, nf)
            # out = self.mlp(vol_feat).squeeze(-1).view(c_shape).unsqueeze(0).unsqueeze(0)

            # to save memory
            out = []
            batch_size = 128 ** 3 # hard coded
            N = coords.shape[0]
            for j in range(N // batch_size + 1):
                coords_ = coords[j * batch_size : (j + 1) * batch_size].unsqueeze(0).unsqueeze(0) # (1, 1, N, 3)
                sample_yz_feat = F.grid_sample(yz_feat, coords_[..., [1, 2]].flip(-1), align_corners=True).permute(0, 2, 3, 1).squeeze(0).squeeze(0)
                sample_xz_feat = F.grid_sample(xz_feat, coords_[..., [0, 2]].flip(-1), align_corners=True).permute(0, 2, 3, 1).squeeze(0).squeeze(0)
                sample_xy_feat = F.grid_sample(xy_feat, coords_[..., [0, 1]].flip(-1), align_corners=True).permute(0, 2, 3, 1).squeeze(0).squeeze(0)

                vol_feat = torch.cat([sample_yz_feat, sample_xz_feat, sample_xy_feat], dim=-1)
                # vol_feat = torch.cat([sample_yz_feat[j * batch_size : (j + 1) * batch_size], 
                #                       sample_xz_feat[j * batch_size : (j + 1) * batch_size],
                #                       sample_xy_feat[j * batch_size : (j + 1) * batch_size]], dim=-1)
                batch_out = self.mlp(vol_feat)
                out.append(batch_out)

            out = torch.cat(out, dim=0).squeeze(-1).view(c_shape).unsqueeze(0).unsqueeze(0)
            
            # sample_yz_feat = F.grid_sample(yz_feat, coords[:, 0, :, :, [1, 2]]) # (1, nf, W, D)
            # sample_xz_feat = F.grid_sample(xz_feat, coords[:, :, 0, :, [0, 2]]) # (1, nf, H, D)
            # sample_xy_feat = F.grid_sample(xy_feat, coords[:, :, :, 0, [0, 1]]) # (1, nf, H, W)

        out = torch.sigmoid(out)
        return out
    
    def forward_head(self, init_noise, init_inp):
        ni = init_noise + init_inp

        # extract triplane features at head
        in_shape = ni.shape[-3:]
        yz_feat = F.adaptive_avg_pool3d(ni, (self.avg_dim, in_shape[1], in_shape[2])).squeeze(1)
        xz_feat = F.adaptive_avg_pool3d(ni, (in_shape[0], self.avg_dim, in_shape[2])).squeeze(1).permute(0, 2, 1, 3)
        xy_feat = F.adaptive_avg_pool3d(ni, (in_shape[0], in_shape[1], self.avg_dim)).squeeze(1).permute(0, 3, 1, 2)
        yz_feat, xz_feat, xy_feat = self.head_conv([yz_feat, xz_feat, xy_feat])
        return [yz_feat, xz_feat, xy_feat]

    def forward_stage(self, tri_feats, i, up_size, tri_noises, mode, coords=None, decode=False):
        yz_feat, xz_feat, xy_feat = tri_feats
        if i > 0:
            # upsample
            # up_size = real_shapes[i]
            yz_feat = F.interpolate(yz_feat, size=(up_size[1], up_size[2]), mode='bilinear', align_corners=True) # .detach()
            xz_feat = F.interpolate(xz_feat, size=(up_size[0], up_size[2]), mode='bilinear', align_corners=True) # .detach()
            xy_feat = F.interpolate(xy_feat, size=(up_size[0], up_size[1]), mode='bilinear', align_corners=True) # .detach()

        yz_feat_prev = yz_feat
        xz_feat_prev = xz_feat
        xy_feat_prev = xy_feat
        
        # add noise, assume rec mode add zero noise
        if i > 0 and mode != 'rec':
            yz_feat = yz_feat + tri_noises[0]
            xz_feat = xz_feat + tri_noises[1]
            xy_feat = xy_feat + tri_noises[2]

        if self.pad_head:
            yz_feat = self.pad_block(yz_feat)
            xz_feat = self.pad_block(xz_feat)
            xy_feat = self.pad_block(xy_feat)

        yz_feat, xz_feat, xy_feat = self.body[i]([yz_feat, xz_feat, xy_feat])
        # skip connect
        if i > 0:
            yz_feat = yz_feat + yz_feat_prev
            xz_feat = xz_feat + xz_feat_prev
            xy_feat = xy_feat + xy_feat_prev
        
        if not decode:
            return [yz_feat, xz_feat, xy_feat]

        out = self.query([yz_feat, xz_feat, xy_feat], coords)
        return out

    def draw_feats(self, init_noise, init_inp, real_shapes, noises_list, mode, end_scale):
        tri_feats = self.forward_head(init_noise, init_inp)

        for i in range(end_scale):
            tri_feats = self.forward_stage(tri_feats, i, real_shapes[i], noises_list[i], mode)
        return tri_feats
    
    def decode_feats(self, tri_feats, real_shapes, noises_list, mode, start_scale, end_scale=-1):
        if end_scale == -1:
            end_scale = len(self.body)
        for i in range(end_scale - start_scale):
            tri_feats = self.forward_stage(tri_feats, start_scale + i, real_shapes[i], noises_list[i], mode)
        out = self.query(tri_feats)
        return out

    def forward(self, init_noise, init_inp, real_shapes, noises_list, mode, coords=None, return_each=False, return_feat=False):
        """_summary_

        Args:
            init_noise (_type_): (1, 1, H, W, D)
            init_inp (_type_): (1, 1, H, W, D)
            real_shapes (_type_): list of (H, W, D)
            mode (str): 
            coords (_type_, optional): _description_. Defaults to None.
            return_each (bool): Defaults to False.

        Returns:
            _type_: _description_
        """
        tri_feats = self.forward_head(init_noise, init_inp)

        out_list = []

        for i, block in enumerate(self.body[:len(real_shapes)]):
            tri_feats = self.forward_stage(tri_feats, i, real_shapes[i], noises_list[i], mode)

            if return_each:
                out = self.query(tri_feats, coords)
                out_list.append(out)

        if return_each:
            return out_list
        
        out = self.query(tri_feats, coords)
        if return_feat:
            return out, tri_feats
        return out


if __name__ == '__main__':
    import time
    net = GrowingGeneratorTriplane(feat_dim=32, pad_head=True)
    s = 16
    init_noise = torch.randn(1, 1, s, s, s).cuda()
    init_inp = torch.zeros(1, 1, s, s, s).cuda()
    real_shapes = []
    for i in range(5):
        net.init_next_stage(0.1)
        s_ = int(s * (4 / 3) ** i)
        real_shapes.append((s_, s_, s_))
    # net.cuda()
    print(net)
