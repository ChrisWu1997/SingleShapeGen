import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import h5py


def load_data_fromH5(path, smooth=True, only_last=False):
    voxel_list = []
    with h5py.File(path, 'r') as fp:
        n_scales = fp.attrs['n_scales']
        for i in range(n_scales):
            voxel = fp[f'scale{i}'][:].astype(np.float)
            if only_last:
                voxel = fp[f'scale{n_scales - 1}'][:]
                return voxel

            if smooth:
                voxel = gaussian_filter(voxel, sigma=0.5)
                voxel = np.clip(voxel, 0.0, 1.0)
            voxel_list.append(voxel)
    
    if voxel_list[0].shape[0] > voxel_list[1].shape[0]:
        voxel_list = voxel_list[::-1]

    return voxel_list


def test():
    pass


if __name__ == "__main__":
    test()
