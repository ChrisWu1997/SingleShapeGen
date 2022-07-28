import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBlock, TriplaneConvs, Convs3DSkipAdd, make_mlp


def get_network(config, name):
    """get specificed network

    Args:
        config (Config): a config object
        name (str): "G" for generator, "D" for discriminator

    Returns:
        network (nn.Module)
    """
    if name == "G":
        if config.G_struct == "triplane":
            return GrowingGeneratorTriplane(config.G_nc, config.G_layers, config.pool_dim, 
                config.feat_dim, config.use_norm, config.mlp_dim, config.mlp_layers)
        elif config.G_struct == "conv3d":
            return GrowingGenerator3D(config.G_nc, config.G_layers, config.use_norm)
        else:
            raise NotImplementedError
    elif name == "D":
        return WDiscriminator(config.D_nc, config.D_layers, config.use_norm)
    else:
        raise NotImplementedError


class WDiscriminator(nn.Module):
    def __init__(self, n_channels=32, n_layers=3, use_norm=True):
        """A 3D convolutional discriminator. 
            Each layer's kernel size, stride and padding size are fixed.

        Args:
            n_channels (int, optional): number of channels for each layer. Defaults to 32.
            n_layers (int, optional): number of conv layers. Defaults to 3.
            use_norm (bool, optional): use normalization layer. Defaults to True.
        """
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
        """A multi-scale generator on tri-plane representation.

        Args:
            n_channels (int, optional): number of channels. Defaults to 32.
            n_layers (int, optional): number of conv layers. Defaults to 4.
            pool_dim (int, optional): average pooling dimension at head. Defaults to 8.
            feat_dim (int, optional): tri-plane feature dimension. Defaults to 32.
            use_norm (bool, optional): use normalization layer. Defaults to True.
            mlp_dim (int, optional): mlp hidden layer feature dimension. Defaults to 32.
            mlp_layers (int, optional): number of mlp hidden layers. Defaults to 0.
        """
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
        """current number of scales"""
        return len(self.body)

    def init_next_scale(self):
        """initialize next scale, i.e., append a conv block"""
        out_c_list = [self.n_channels] * (self.n_layers - 1) + [self.feat_dim]
        ker_size, stride, pad = 3, 1, 1 # hard-coded
        model = TriplaneConvs(self.feat_dim, out_c_list, ker_size, stride, pad, self.use_norm)
        self.body.append(model)
    
    def query(self, tri_feats: list, coords=None):
        """construct output volume through point quries.

        Args:
            tri_feats (list): tri-plane feature maps, [yz_feat, xz_feat, xy_feat]
            coords (tensor, optional): query points of shape (H, W, D, 3). If None, use the size of tri_feats.
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
    
    def forward_head(self, init_noise):
        """forward through the projection module at head."""
        ni = init_noise
        # extract triplane features at head
        in_shape = ni.shape[-3:]
        yz_feat = F.adaptive_avg_pool3d(ni, (self.pool_dim, in_shape[1], in_shape[2])).squeeze(1)
        xz_feat = F.adaptive_avg_pool3d(ni, (in_shape[0], self.pool_dim, in_shape[2])).squeeze(1).permute(0, 2, 1, 3)
        xy_feat = F.adaptive_avg_pool3d(ni, (in_shape[0], in_shape[1], self.pool_dim)).squeeze(1).permute(0, 3, 1, 2)
        yz_feat, xz_feat, xy_feat = self.head_conv([yz_feat, xz_feat, xy_feat])
        return [yz_feat, xz_feat, xy_feat]

    def forward_scale(self, tri_feats: list, i: int, up_size: list, tri_noises: list, mode: str, coords=None, decode=False):
        """forward through the generator block at scale i.

        Args:
            tri_feats (list): tri-plane feature maps
            i (int): i-th scale 
            up_size (list): upsampled size
            tri_noises (list): tri-plane noise maps
            mode (str): "rec" for ignoring noise
            coords (tensor, optional): query point coordinates. Defaults to None.
            decode (bool, optional): decode output volume. Defaults to False.

        Returns:
            output: if decode, generated shape volume; else, tri-plane feature maps.
        """
        yz_feat, xz_feat, xy_feat = tri_feats
        if i > 0:
            # upsample
            # up_size = real_sizes[i]
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

    def draw_feats(self, init_noise: list, real_sizes: list, noises_list: list, mode: str, end_scale: int):
        """draw generated tri-plane feature maps at end_scale. To facilitate training."""
        tri_feats = self.forward_head(init_noise)

        for i in range(end_scale):
            tri_feats = self.forward_scale(tri_feats, i, real_sizes[i], noises_list[i], mode)
        return tri_feats
    
    def decode_feats(self, tri_feats: list, real_sizes: list, noises_list: list, mode: str, start_scale: int, end_scale=-1):
        """pass tri-plane feature maps from start_scale to end_scale, and decode output volume. 
            To facilitate training."""
        if end_scale == -1:
            end_scale = len(self.body)
        for i in range(end_scale - start_scale):
            tri_feats = self.forward_scale(tri_feats, start_scale + i, real_sizes[i], noises_list[i], mode)
        out = self.query(tri_feats)
        return out

    def forward(self, init_noise: torch.Tensor, real_sizes: list, noises_list: list, mode: str, coords=None, return_each=False, return_feat=False):
        """forward through the model

        Args:
            init_noise (torch.Tensor): input 3D noise tensor
            real_sizes (list): list of multi-scale shape sizes
            noises_list (list): list of multi-scale tri-plane noises
            mode (str): "rand" or "rec"
            coords (torch.Tensor, optional): query point coordinates. Defaults to None.
            return_each (bool, optional): return output at each scale. Defaults to False.
            return_feat (bool, optional): return also feature maps. Defaults to False.

        Returns:
            output: 3D shape volume, or a list of 3D shape volume, or feature maps
        """
        tri_feats = self.forward_head(init_noise)

        out_list = []

        for i, block in enumerate(self.body[:len(real_sizes)]):
            tri_feats = self.forward_scale(tri_feats, i, real_sizes[i], noises_list[i], mode)

            if return_each:
                out = self.query(tri_feats, coords)
                out_list.append(out)

        if return_each:
            return out_list
        
        out = self.query(tri_feats, coords)
        if return_feat:
            return out, tri_feats
        return out


class GrowingGenerator3D(nn.Module):
    def __init__(self, n_channels=32, n_layers=4, use_norm=True, pad_head=False):
        """A multi-scale generator on tri-plane representation.

        Args:
            n_channels (int, optional): number of channels. Defaults to 32.
            n_layers (int, optional): number of conv layers. Defaults to 4.
            use_norm (bool, optional): use normalization layer. Defaults to True.
            pad_head (bool, optional): make zero padding at head. Defaults to False.
        """
        super(GrowingGenerator3D, self).__init__()
        self.nf = n_channels
        self.n_conv_layers = n_layers
        self.use_norm = use_norm
        self.pad_len = 0 if pad_head else 1
        
        self.body = nn.ModuleList([])

    @property
    def n_scales(self):
        """current number of scales"""
        return len(self.body)

    def init_next_scale(self):
        """initialize next scale, i.e., append a conv block"""
        model = Convs3DSkipAdd(self.nf, self.n_conv_layers, 3, 1, self.use_norm)
        self.body.append(model)

    def forward(self, inp: torch.Tensor, noises_list: list, start_scale=0, end_scale=-1, return_each=False):
        """forward through the model

        Args:
            inp (torch.Tensor): input 3D volume
            noises_list (list): list of 3D noise
            start_scale (int, optional): start scale. Defaults to 0.
            end_scale (int, optional): end scale. Defaults to -1.
            return_each (bool, optional): return output at each scale. Defaults to False.

        Returns:
            output: a 3D shape volume, or a list of 3D shape volume
        """
        out_list = []
        out = inp
        if end_scale == -1:
            end_scale = len(self.body) - 1
        if len(noises_list) != end_scale + 1 - start_scale:
            raise RuntimeError

        for i, block in enumerate(self.body[start_scale:end_scale + 1]):
            if i + start_scale > 0:
                out = F.interpolate(out, size=noises_list[i].shape[-3:], mode='trilinear', align_corners=True)
            out = block(noises_list[i], out)

            if return_each:
                out_list.append(out)

        if return_each:
            return out_list

        return out
