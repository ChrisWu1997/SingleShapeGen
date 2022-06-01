import numpy as np
import torch


def calc_gradient_penalty(netD, real_data, fake_data):
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


def set_require_grads(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)


def generate_tri_plane_noise(res_x, res_y, res_z, nf, noise_amp, device):
    noise = [(torch.randn(1, nf, res_y, res_z, device=device) * noise_amp).detach(), 
             (torch.randn(1, nf, res_x, res_z, device=device) * noise_amp).detach(), 
             (torch.randn(1, nf, res_x, res_y, device=device) * noise_amp).detach()]
    return noise


def make_coord(H, W, D, device, normalize=True):
    """ Make coordinates at grid centers."""
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
    img1 = volume[volume.shape[0] // 2]
    img2 = volume[:, volume.shape[0] // 2]
    img3 = volume[:, :, volume.shape[0] // 2]
    _max = max(img1.shape[0], img2.shape[0], img3.shape[0])

    img1 = np.pad(img1, [(0, _max - img1.shape[0]), (1, 1)])
    img2 = np.pad(img2, [(0, _max - img2.shape[0]), (1, 1)])
    img3 = np.pad(img3, [(0, _max - img3.shape[0]), (1, 1)])

    img = np.concatenate([img1, img2, img3], axis=1)
    return img


class TrainClock(object):
    """ Clock object to track epoch and step during training
    """
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
