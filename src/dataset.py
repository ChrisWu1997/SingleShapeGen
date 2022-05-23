import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from torch.nn.functional import interpolate
import trimesh
from trimesh.voxel.creation import voxelize
import h5py


def generate_3Ddata_multiscale(config):
    print("load from pre-voxelized")
    ret_list = load_voxel_fromh5(config.path, config.n_scales)

    ret_list = [torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda() for x in ret_list]
    return ret_list


def load_mesh(path):
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        print("convert trimesh.Scene to trimesh.Trimeh")
        geo_list = []
        for i, g in enumerate(mesh.geometry.values()):
            geo_list.append(g)
        mesh = trimesh.util.concatenate(geo_list)

    mesh.fix_normals(multibody=True)
    return mesh


def load_voxel_fromh5(path, n_scales, smooth=True):
    voxel_list = []
    with h5py.File(path, 'r') as fp:
        for i in range(n_scales):
            voxel = fp[f'scale{i}'][:].astype(np.float)
            if smooth:
                voxel = gaussian_filter(voxel, sigma=0.5)
                voxel = np.clip(voxel, 0.0, 1.0)
            voxel_list.append(voxel)
    
    if voxel_list[0].shape[0] > voxel_list[1].shape[0]:
        voxel_list = voxel_list[::-1]
    template = gaussian_filter(voxel_list[0], sigma=2.0)
    voxel_list = [template] + voxel_list

    return voxel_list


def test():
    pass


if __name__ == "__main__":
    test()
