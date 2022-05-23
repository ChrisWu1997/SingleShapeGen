import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import zoom
import torch
from torch.nn.functional import interpolate
import torch.nn.functional as F
import random
import trimesh
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from trimesh.voxel.creation import voxelize
import h5py


def to_binary(tensor):
    return torch.where(tensor < 0.5, torch.zeros_like(tensor), torch.ones_like(tensor))


def _pad_img(img, dim):
    img_ = np.zeros((dim, img.shape[1]))
    img_[(dim - img.shape[0]) // 2:(dim - img.shape[0]) // 2 + img.shape[0], :] = img
    return img_


def project_along_xyz(voxel, concat=True, mid=True):
    img1 = np.amax(voxel, axis=0) if not mid else voxel[voxel.shape[0] // 2]
    img2 = np.amax(voxel, axis=1) if not mid else voxel[:, voxel.shape[1] // 2]
    img3 = np.amax(voxel, axis=2) if not mid else voxel[:, :, voxel.shape[2] // 2]
    if concat:
        dim = max(img1.shape[0], img2.shape[0], img3.shape[0])
        line = np.zeros((dim, 2))
        whole_img = np.concatenate([_pad_img(img1, dim), line, _pad_img(img2, dim), line, _pad_img(img3, dim)], axis=1)
        return whole_img
    else:
        return img1, img2, img3


def draw_mat_figure_along_xyz(voxel, mid=True, grayscale=True):
    img1 = np.amax(voxel, axis=0) if not mid else voxel[voxel.shape[0] // 2]
    img2 = np.amax(voxel, axis=1) if not mid else voxel[:, voxel.shape[1] // 2]
    img3 = np.amax(voxel, axis=2) if not mid else voxel[:, :, voxel.shape[2] // 2]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 3))
    cax1 = ax1.matshow(img1, cmap='gray') if grayscale else ax1.matshow(img1)
    cax2 = ax2.matshow(img2, cmap='gray') if grayscale else ax2.matshow(img2)
    cax3 = ax3.matshow(img3, cmap='gray') if grayscale else ax3.matshow(img3)
    fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
    fig.colorbar(cax3, ax=ax3, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def zoom_torch(tensor, scale):
    tensor = tensor.detach().cpu().numpy()
    tensor = zoom(tensor, scale)
    tensor = torch.from_numpy(tensor).type(torch.float32).cuda()
    return tensor


def interpolate_torch(tensor, scale=None, size=None):
    assert len(tensor.shape) == 5
    if size is not None:
        assert len(size) == 3
        tensor = interpolate(tensor, size=size, mode='trilinear', align_corners=True)
    elif scale is not None:
        assert len(scale) == 3
        tensor = interpolate(tensor, scale_factor=scale, mode='trilinear', align_corners=True)
    return tensor


def nonempty_coords(voxel, threshold=0.1):
    mask = voxel > threshold
    mask = mask.squeeze(0).squeeze(0)
    coords = torch.nonzero(mask)
    return coords, mask


def nonempty_surf_coords(voxel, threshold=0.1, receptive_size=11, padding=True):
    """[summary]

    Args:
        voxel ([type]): (1, 1, H, W, D)
        threshold (float, optional): [description]. Defaults to 0.1.
        receptive_size (int, optional): [description]. Defaults to 11.

    Returns:
        coords: (N, 3)
        mask: (H, W, D) boolean
    """
    # filter_ = torch.ones((1, 1, receptive_size, receptive_size, receptive_size), dtype=torch.float32, device=voxel.device)
    if padding:
        voxel = F.pad(voxel, [receptive_size // 2] * 6, mode='replicate')
    # voxel_filtered = F.conv3d(voxel, filter_) / (receptive_size ** 3)
    voxel_filtered = F.avg_pool3d(voxel, kernel_size=receptive_size, stride=1, padding=0)
    mask = torch.logical_and(voxel_filtered >= threshold, voxel_filtered <= 1 - threshold)

    # mask = voxel > threshold
    # mask = torch.logical_and(mask, voxel > threshold)
    mask = mask.squeeze(0).squeeze(0)
    coords = torch.nonzero(mask)
    if not padding:
        coords = coords + receptive_size // 2
    return coords, mask


def nonempty_surf_coords_more(voxel, patch_size=5, padding=True):
    if padding:
        voxel = F.pad(voxel, [patch_size // 2] * 6, mode='replicate')
    voxel_filtered = F.avg_pool3d(voxel, kernel_size=patch_size, stride=1, padding=0)
    threshold = 1. / patch_size ** 3
    mask = torch.logical_and(voxel_filtered >= threshold, voxel_filtered <= 1 - threshold)

    mask = mask.squeeze(0).squeeze(0)
    coords = torch.nonzero(mask)
    if not padding:
        coords = coords + patch_size // 2
    return coords, mask


def sample_nonempty_surf_patch(voxel, k, threshold=0.1, patch_size=11, padding=False, centers=None, return_center=False):
    """voxel: (1, 1, H, W, D) """
    gap = patch_size // 2
    if centers is None:
        coords_, mask = nonempty_surf_coords(voxel, threshold, patch_size, padding)
        # coords_, mask = nonempty_surf_coords_more(voxel, patch_size, padding)

        if coords_.shape[0] == 0:
            print("no valid patches, randomly select in the whole space.")
            H, W, D = voxel.shape[-3:]
            centers = torch.stack([torch.randint(gap, H - gap - 1, size=(k, )),
                                torch.randint(gap, W - gap - 1, size=(k, )), 
                                torch.randint(gap, D - gap - 1, size=(k, ))], dim=-1).to(voxel.device)
        else:
            k = min(k, coords_.shape[0])
            indices = random.sample(range(coords_.shape[0]), k)
            centers = coords_[indices] # (k, 3)

    lin = torch.linspace(-gap, -gap + patch_size - 1, patch_size, dtype=torch.long, device=voxel.device)
    grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing='ij')
    points = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    points = points.unsqueeze(0).expand(k, -1, -1, -1, -1) + centers.view(k, 1, 1, 1, 3) # (K, M, M, M, 3)
    points = points.view(-1, 3) # (K * M ** 3, 3)

    patches = voxel[0, 0, points[:, 0], points[:, 1], points[:, 2]] # (K * M ** 3)
    patches = patches.view(k, patch_size, patch_size, patch_size).unsqueeze(1)
    if return_center:
        return patches, centers.detach()
    return patches


def sample_random_patches(voxel, k, patch_size, pos=None, return_pos=False, pad=True):
    if pad is True:
        l = patch_size // 2
        voxel_p = F.pad(voxel, (l, l, l, l, l, l))
    else:
        voxel_p = voxel

    H, W, D = voxel_p.shape[-3:]
    patch_list = []
    pos_list = []
    for i in range(k):
        if pos is None:
            sx = random.randint(0, H - patch_size)
            sy = random.randint(0, W - patch_size)
            sz = random.randint(0, D - patch_size)
        else:
            sx, sy, sz = pos[i]
        patch = voxel_p[:, :, sx:sx+patch_size, sy:sy+patch_size, sz:sz+patch_size]
        patch_list.append(patch)
        pos_list.append((sx, sy, sz))
    patch_list = torch.cat(patch_list, dim=0)
    if return_pos:
        return patch_list, pos_list
    return patch_list


def sample_random_patch_coords(voxel_shape, k, patch_size, device, pos=None, return_pos=False):
    H, W, D = voxel_shape
    patch_list = []
    pos_list = []
    for i in range(k):
        if pos is None:
            sx = random.randint(0, H - patch_size)
            sy = random.randint(0, W - patch_size)
            sz = random.randint(0, D - patch_size)
        else:
            sx, sy, sz = pos[i]
        xx = torch.arange(sx, sx + patch_size, device=device)
        yy = torch.arange(sy, sy + patch_size, device=device)
        zz = torch.arange(sz, sz + patch_size, device=device)
        coords = torch.stack(torch.meshgrid([xx, yy, zz], indexing='ij'), dim=-1)
        patch_list.append(coords)
        pos_list.append((sx, sy, sz))
    patch_list = torch.stack(patch_list, dim=0) # (B, H, W, D, 3)
    if return_pos:
        return patch_list, pos_list
    return patch_list


if __name__ == '__main__':
    import time
    s = 128
    inp = torch.zeros(1, 1, s, s, s).cuda()
    inp[:, :, 0:64, 0:64, 0:64] = 1
    since = time.time()
    patches = sample_nonempty_surf_patch(inp, k=64, patch_size=15)
    print(patches.shape)
    print(time.time() - since)
