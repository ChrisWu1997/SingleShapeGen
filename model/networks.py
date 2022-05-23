import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBlock, TriplaneConvs, make_mlp


def get_network(config, name):
    if name == 'G':
        return GrowingGeneratorTriplane(config.G_nc, config.G_layers, config.pool_dim, 
            config.feat_dim, config.use_norm, config.mlp_dim, config.mlp_layers)
    elif name == 'D':
        return WDiscriminator(config.D_nc, config.D_layers, config.use_norm)
    else:
        raise NotImplementedError


class WDiscriminator(nn.Module):
    def __init__(self, n_channels=32, n_layers=3, use_norm=True):
        super(WDiscriminator, self).__init__()
        ker_size, stride, pad = 3, 2, 1 # hard-coded
        self.head = ConvBlock(1, n_channels, ker_size, stride, pad, use_norm, sdim='3d')

        self.body = nn.Sequential()
        for i in range(n_layers - 2):
            ker_size, stride, pad = 3, 1, 0 # hard-coded
            block = ConvBlock(n_channels, n_channels, ker_size, stride, pad, use_norm, sdim='3d')
            self.body.add_module('block%d' % (i + 1), block)

        ker_size, stride, pad = 3, 1, 0 # hard-coded
        self.tail = nn.Conv3d(n_channels, 1, ker_size, stride, pad)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GrowingGeneratorTriplane(nn.Module):
    def __init__(self, n_channels=32, n_layers=4, pool_dim=8, feat_dim=32, use_norm=True, mlp_dim=32, mlp_layers=0):
        super(GrowingGeneratorTriplane, self).__init__()
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.pool_dim = pool_dim
        self.feat_dim = feat_dim
        self.use_norm = use_norm
        
        # 1x1 conv
        self.head_conv = TriplaneConvs(self.pool_dim, self.feat_dim, 1, 1, 0, False)
        
        self.body = nn.ModuleList([])
        
        self.mlp = make_mlp(feat_dim * 3, 1, mlp_dim, mlp_layers) # TODO: add option for feature sum

    @property
    def n_scales(self):
        return len(self.body)

    def init_next_scale(self):
        out_c_list = [self.n_channels] * (self.n_layers - 1) + [self.feat_dim]
        ker_size, stride, pad = 3, 1, 1 # hard-coded
        model = TriplaneConvs(self.feat_dim, out_c_list, ker_size, stride, pad, self.use_norm)
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
            # FIXME: should assume coords to be (N, 3)
            # coords shape: (H, W, D, 3)
            c_shape = coords.shape[:3]
            coords = coords.view(-1, 3)
            # to save memory
            out = []
            batch_size = 128 ** 3 # FIXME: hard coded, prevent overflow
            N = coords.shape[0]
            for j in range(N // batch_size + 1):
                coords_ = coords[j * batch_size : (j + 1) * batch_size].unsqueeze(0).unsqueeze(0) # (1, 1, N, 3)
                sample_yz_feat = F.grid_sample(yz_feat, coords_[..., [1, 2]].flip(-1), align_corners=True).permute(0, 2, 3, 1).squeeze(0).squeeze(0)
                sample_xz_feat = F.grid_sample(xz_feat, coords_[..., [0, 2]].flip(-1), align_corners=True).permute(0, 2, 3, 1).squeeze(0).squeeze(0)
                sample_xy_feat = F.grid_sample(xy_feat, coords_[..., [0, 1]].flip(-1), align_corners=True).permute(0, 2, 3, 1).squeeze(0).squeeze(0)

                vol_feat = torch.cat([sample_yz_feat, sample_xz_feat, sample_xy_feat], dim=-1)
                batch_out = self.mlp(vol_feat)
                out.append(batch_out)

            out = torch.cat(out, dim=0).squeeze(-1).view(c_shape).unsqueeze(0).unsqueeze(0)

        out = torch.sigmoid(out)
        return out
    
    def forward_head(self, init_noise, init_inp):
        ni = init_noise + init_inp

        # extract triplane features at head
        in_shape = ni.shape[-3:]
        yz_feat = F.adaptive_avg_pool3d(ni, (self.pool_dim, in_shape[1], in_shape[2])).squeeze(1)
        xz_feat = F.adaptive_avg_pool3d(ni, (in_shape[0], self.pool_dim, in_shape[2])).squeeze(1).permute(0, 2, 1, 3)
        xy_feat = F.adaptive_avg_pool3d(ni, (in_shape[0], in_shape[1], self.pool_dim)).squeeze(1).permute(0, 3, 1, 2)
        yz_feat, xz_feat, xy_feat = self.head_conv([yz_feat, xz_feat, xy_feat])
        return [yz_feat, xz_feat, xy_feat]

    def forward_scale(self, tri_feats, i, up_size, tri_noises, mode, coords=None, decode=False):
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

        # if self.pad_head:
        #     yz_feat = self.pad_block(yz_feat)
        #     xz_feat = self.pad_block(xz_feat)
        #     xy_feat = self.pad_block(xy_feat)

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
            tri_feats = self.forward_scale(tri_feats, i, real_shapes[i], noises_list[i], mode)
        return tri_feats
    
    def decode_feats(self, tri_feats, real_shapes, noises_list, mode, start_scale, end_scale=-1):
        if end_scale == -1:
            end_scale = len(self.body)
        for i in range(end_scale - start_scale):
            tri_feats = self.forward_scale(tri_feats, start_scale + i, real_shapes[i], noises_list[i], mode)
        out = self.query(tri_feats)
        return out

    def forward(self, init_noise, init_inp, real_shapes, noises_list, mode, coords=None, return_each=False, return_feat=False):
        # FIXME: remove init_inp, as it's always 0
        tri_feats = self.forward_head(init_noise, init_inp)

        out_list = []

        for i, block in enumerate(self.body[:len(real_shapes)]):
            tri_feats = self.forward_scale(tri_feats, i, real_shapes[i], noises_list[i], mode)

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
    load_path = "/local/cg/rundi/workspace/ssg_code/project_log/vtpfmsV1_acropolisFm15_res128s6/model/scale5_latest.pth"
    checkpoint = torch.load(load_path)
    
    netG = GrowingGeneratorTriplane()
    n_scale = 6
    for s in range(n_scale):
        netG.init_next_stage()
    netG.load_state_dict(checkpoint['netG_state_dict'])
    print(netG)

    netD = WDiscriminator()
    netD.load_state_dict(checkpoint['netD_state_dict'])
    print(netD)
