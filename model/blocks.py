import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ker_size,
                 stride,
                 padd,
                 use_norm=True,
                 sdim='3d',
                 padd_mode='zeros',
                 onlyconv=False):
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

    def __init__(self,
                 in_channels,
                 out_channels_list,
                 ker_size=3,
                 stride=1,
                 padd=0,
                 use_norm=True):
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

    def make_2Dconvs(self, in_c, out_c_list, ker_size, stride, padd, use_norm):
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

    def forward(self, tri_feats):
        yz_feat = self.conv_yz(tri_feats[0])
        xz_feat = self.conv_xz(tri_feats[1])
        xy_feat = self.conv_xy(tri_feats[2])
        return yz_feat, xz_feat, xy_feat


def make_mlp(in_features,
             out_features,
             hidden_features,
             n_hidden_layers,
             act=nn.ReLU(inplace=True)):
    layer_list = [nn.Linear(in_features, hidden_features), act]
    for i in range(n_hidden_layers):
        layer_list.extend([nn.Linear(hidden_features, hidden_features), act])
    layer_list.extend([nn.Linear(hidden_features, out_features)])
    return nn.Sequential(*layer_list)


if __name__ == '__main__':

    mlp2 = make_mlp(32, 1, 32, 1)
    print(mlp2)
