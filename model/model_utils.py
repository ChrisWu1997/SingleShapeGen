import numpy as np
import torch


def calc_gradient_penalty(netD, real_data: torch.Tensor, fake_data: torch.Tensor, device: torch.device):
    """calculate gradient penalty (WGAN-GP), average over sptial dimensions.

    Args:
        netD (nn.Module): discriminator (critic) network
        real_data (torch.Tensor): real (reference) data
        fake_data (torch.Tensor): fake (generated) data

    Returns:
        gradient penalty: torch.Tensor
    """
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device) #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() # average over sptial dimensions
    return gradient_penalty


def set_require_grads(model, require_grad: bool):
    """set requires_grad for model parameters"""
    for p in model.parameters():
        p.requires_grad_(require_grad)


def generate_tri_plane_noise(res_x: int, res_y: int, res_z: int, nf: int, noise_amp: float, device: torch.device):
    """generate a tuple of added noise for tri-plane maps.

    Args:
        res_x (int): x-axis resolution
        res_y (int): y-axis resolution
        res_z (int): z-axis resolution
        nf (int): number of features
        noise_amp (float): noise std
        device (torch.device): much quicker if directly generating on device

    Returns:
        noise maps: three noise tensors
    """
    noise = [(torch.randn(1, nf, res_y, res_z, device=device) * noise_amp).detach(), 
             (torch.randn(1, nf, res_x, res_z, device=device) * noise_amp).detach(), 
             (torch.randn(1, nf, res_x, res_y, device=device) * noise_amp).detach()]
    return noise


def generate_3d_noise(res_x: int, res_y: int, res_z: int, noise_amp: float, device: torch.device):
    """generate a tuple of added 3D noise."""
    noise = (torch.randn(1, 1, res_x, res_y, res_z, device=device) * noise_amp).detach()
    return noise


def make_coord(H: int, W: int, D: int, device: torch.Tensor, normalize=True):
    """Generate xyz coordinates for all points at grid centers.

    Args:
        H (int): height
        W (int): width
        D (int): depth
        device (torch.Tensor): torch device
        normalize (bool, optional): normalize to [-1, 1]. Defaults to True.

    Returns:
        torch.Tensor: point coordinates of shape (H, W, D, 3)
    """
    xs = torch.arange(H, device=device).float() 
    ys = torch.arange(W, device=device).float()
    zs = torch.arange(D, device=device).float()
    if normalize:
        xs = xs / (H - 1) * 2 - 1 # (-1, 1)
        ys = ys / (W - 1) * 2 - 1 # (-1, 1)
        zs = zs / (D - 1) * 2 - 1 # (-1, 1)

    coords = torch.stack(torch.meshgrid(xs, ys, zs), dim=-1)
    return coords


def slice_volume_along_xyz(volume: np.ndarray):
    """slice a 3D volume along the mid point of each axis"""
    img1 = volume[volume.shape[0] // 2]
    img2 = volume[:, volume.shape[1] // 2]
    img3 = volume[:, :, volume.shape[2] // 2]
    _max = max(img1.shape[0], img2.shape[0], img3.shape[0])

    img1 = np.pad(img1, [(0, _max - img1.shape[0]), (1, 1)])
    img2 = np.pad(img2, [(0, _max - img2.shape[0]), (1, 1)])
    img3 = np.pad(img3, [(0, _max - img3.shape[0]), (1, 1)])

    img = np.concatenate([img1, img2, img3], axis=1)
    return img


class TrainClock(object):
    """ Clock object to track epoch and step during training"""
    def __init__(self):
        self.epoch = 1
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def reset(self):
        self.epoch = 1
        self.minibatch = 0
        self.step = 0

    def make_checkpoint(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'step': self.step
        }

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']
