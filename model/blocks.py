import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, ker_size: int, stride: int, padd: int,
                use_norm=True, sdim='3d', padd_mode='zeros', onlyconv=False):
        """Basic Conv-IN-LReLU block.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            ker_size (int): kernel size
            stride (int): stride
            padd (int): padding size
            use_norm (bool, optional): use normalization layer. Defaults to True.
            sdim (str, optional): spatial dimension (3d or 2d). Defaults to '3d'.
            padd_mode (str, optional): padding mode. Defaults to 'zeros'.
            onlyconv (bool, optional): keep only convolution layer. Defaults to False.
        """
        super(ConvBlock, self).__init__()
        if sdim == '3d':
            self.add_module(
                'conv',
                nn.Conv3d(in_channels,
                          out_channels,
                          kernel_size=ker_size,
                          stride=stride,
                          padding=padd,
                          padding_mode=padd_mode,
                          bias=use_norm is False))
        else:
            self.add_module(
                'conv',
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=ker_size,
                          stride=stride,
                          padding=padd,
                          padding_mode=padd_mode,
                          bias=use_norm is False))
        if not onlyconv:
            if use_norm:
                if sdim == '3d':
                    self.add_module('norm', nn.InstanceNorm3d(out_channels))
                else:
                    self.add_module('norm', nn.InstanceNorm2d(out_channels))
            self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv3d') != -1 or classname.find('Conv2d') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('Norm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class TriplaneConvs(nn.Module):
    def __init__(self, in_channels: int, out_channels_list: list, ker_size=3, stride=1, padd=0, use_norm=True):
        """Triplane convolution block.

        Args:
            in_channels (int): number of input channels.
            out_channels_list (list): a list of output channel numbers for each layer.
            ker_size (int, optional): kernel size. Defaults to 3.
            stride (int, optional): stride. Defaults to 1.
            padd (int, optional): padding size. Defaults to 0.
            use_norm (bool, optional): use normalization layer. Defaults to True.
        """
        super(TriplaneConvs, self).__init__()
        if type(out_channels_list) is int:
            in_c = in_channels
            out_c = out_channels_list
            self.conv_yz = nn.Conv2d(in_c, out_c, ker_size, stride, padd)
            self.conv_xz = nn.Conv2d(in_c, out_c, ker_size, stride, padd)
            self.conv_xy = nn.Conv2d(in_c, out_c, ker_size, stride, padd)
        else:
            self.conv_yz = self.make_2Dconvs(in_channels, out_channels_list,
                                             ker_size, stride, padd, use_norm)
            self.conv_xz = self.make_2Dconvs(in_channels, out_channels_list,
                                             ker_size, stride, padd, use_norm)
            self.conv_xy = self.make_2Dconvs(in_channels, out_channels_list,
                                             ker_size, stride, padd, use_norm)

    def make_2Dconvs(self, in_c: int, out_c_list: list, ker_size: int, stride: int, padd: int, use_norm: bool):
        """make a sequence of 2D ConvBlock"""
        model = nn.Sequential()
        model.add_module(
            'conv0',
            ConvBlock(in_c, out_c_list[0], ker_size, stride, padd, use_norm,
                      sdim='2d', padd_mode='zeros'))

        for i in range(len(out_c_list) - 1):
            is_last = i == len(out_c_list) - 2
            use_norm = use_norm if not is_last else False
            model.add_module(
                f'conv{i + 1}',
                ConvBlock(out_c_list[i], out_c_list[i + 1], ker_size, stride, padd, use_norm,
                          sdim='2d', padd_mode='zeros', onlyconv=is_last))
        return model

    def forward(self, tri_feats: list, tri_noises: list = None, add_noise=True, skip_add=True):
        """
        Args:
            tri_feats (list): tri-plane feature maps
            tri_noises (list, optional): tri-plane noise maps
            add_noise (bool, optional): add tri-lane noise maps
            skip_add (bool, optional): skip connection

        Returns:
            output: tri-plane feature maps
        """
        yz_feat, xz_feat, xy_feat = tri_feats
        if skip_add:
            yz_feat_prev = yz_feat
            xz_feat_prev = xz_feat
            xy_feat_prev = xy_feat
        
        # add noise, assume rec mode add zero noise
        if add_noise and tri_noises is not None:
            yz_feat = yz_feat + tri_noises[0]
            xz_feat = xz_feat + tri_noises[1]
            xy_feat = xy_feat + tri_noises[2]

        yz_feat = self.conv_yz(yz_feat)
        xz_feat = self.conv_xz(xz_feat)
        xy_feat = self.conv_xy(xy_feat)

        # skip connect
        if skip_add:
            yz_feat = yz_feat + yz_feat_prev
            xz_feat = xz_feat + xz_feat_prev
            xy_feat = xy_feat + xy_feat_prev
        
        return [yz_feat, xz_feat, xy_feat]


class Convs3DSkipAdd(nn.Module):
    def __init__(self, n_channels=32, n_layers=4, ker_size=3, stride=1, use_norm=True, pad_head=False):
        """A sequence of 3D convolutions with skip connection.

        Args:
            n_channels (int, optional): number of channels. Defaults to 32.
            n_layers (int, optional): number of layers. Defaults to 4.
            ker_size (int, optional): kernel size. Defaults to 3.
            stride (int, optional): stride. Defaults to 1.
            use_norm (bool, optional): use normalization layer. Defaults to True.
            pad_head (bool, optional): make zero padding at head. Defaults to False.
        """
        super(Convs3DSkipAdd, self).__init__()
        pad_len = 0 if pad_head else 1
        self.pad_head = pad_head

        self.head = ConvBlock(1, n_channels, ker_size, stride, pad_len, use_norm) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(n_layers - 2):
            block = ConvBlock(n_channels, n_channels, ker_size, stride, pad_len, use_norm)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(
            nn.Conv3d(n_channels, 1, kernel_size=ker_size, stride=stride, padding=pad_len),
            nn.Tanh()
        )

        if pad_head:
            pad_noise = int(((3 - 1) * n_layers) / 2)
            pad_image = int(((3 - 1) * n_layers) / 2)
            self.m_noise = nn.ConstantPad3d(int(pad_noise), 0)
            self.m_image = nn.ConstantPad3d(int(pad_image), 0)

    def forward(self, noise, inp):
        if self.pad_head:
            x = self.m_noise(noise)
            y_ = self.m_image(inp)
        else:
            x = noise
            y_ = inp

        x = self.head(x + y_)
        x = self.body(x)
        x = self.tail(x)
        out = x + inp
        out = torch.max(torch.min(out, out*0.002+0.998), out*0.002)
        return out


def make_mlp(in_features: int, out_features: int, hidden_features: int, n_hidden_layers: int, 
            act=nn.ReLU(inplace=True)):
    """Make a MLP.

    Args:
        in_features (int): number of input features.
        out_features (int): number of output features.
        hidden_features (int): number of hidden layer features.
        n_hidden_layers (int): number of hidden layers.
        act (nn.Module, optional): activation function. Defaults to nn.ReLU(inplace=True).

    Returns:
        MLP (nn.Module)
    """
    layer_list = [nn.Linear(in_features, hidden_features), act]
    for i in range(n_hidden_layers):
        layer_list.extend([nn.Linear(hidden_features, hidden_features), act])
    layer_list.extend([nn.Linear(hidden_features, out_features)])
    return nn.Sequential(*layer_list)
