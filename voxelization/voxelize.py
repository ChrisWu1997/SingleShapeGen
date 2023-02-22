import os
import argparse
import h5py
import numpy as np
import torch
from torch.nn.functional import interpolate
import trimesh
from trimesh.voxel import VoxelGrid
from trimesh.voxel.encoding import DenseEncoding
import binvox_rw


BINVOX_PATH = "./binvox"


def load_mesh(path: str, normalize=True):
    """load mesh file and (optionally) normalize it within unit sphere"""
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        print("convert trimesh.Scene to trimesh.Trimeh")
        geo_list = []
        for i, g in enumerate(mesh.geometry.values()):
            geo_list.append(g)
        mesh = trimesh.util.concatenate(geo_list)

    mesh.fix_normals(multibody=True)

    if normalize:
        verts = mesh.vertices
        centers = np.mean(verts, axis=0)
        verts = verts - centers
        length = np.max(np.linalg.norm(verts, 2, axis=1))
        verts = verts * (1. / length)
        mesh.vertices = verts
    return mesh


def multiscale_voxelization_binvox(path: str, resolution_list: list, min_size: int):
    """voxelize a mesh at different resolution (multi-scale)

    Args:
        path (str): path to mesh file
        resolution_list (list): list of shape resolution (i.e., largest dimension of xyz)
            [(H_0, W_0, D_0), (H_1, W_1, D_1), ...], ascending order
        min_size (int): mininum spatial dimension

    Returns:
        list: a list of multi-scale 3D shape
    """
    # normalize to -1 ~ 1 and save, otherwise binvox might give incorrect result
    mesh = load_mesh(path, normalize=True)
    norm_path = os.path.splitext(path)[0] + '_norm.obj'
    mesh.export(norm_path)
    mesh = None

    binvox_res = resolution_list[0]
    voxel_list = []
    for i in range(len(resolution_list)): # ascending resolution
        if i > 0:
            scale = resolution_list[i] / resolution_list[i - 1]
            binvox_res = np.round(binvox_res * scale)

        # voxelize by binvox (only surface voxels)
        command = f'{BINVOX_PATH} -cb -e -d {binvox_res} {norm_path}'
        os.system(command)
        out_path = os.path.splitext(norm_path)[0] + '.binvox'
        
        # load voxels and fill inside
        with open(out_path, 'rb') as f:
            data = binvox_rw.read_as_3d_array(f).data
        bound_x = np.nonzero(np.sum(data, axis=(1, 2)))[0][[0, -1]]
        bound_y = np.nonzero(np.sum(data, axis=(0, 2)))[0][[0, -1]]
        bound_z = np.nonzero(np.sum(data, axis=(0, 1)))[0][[0, -1]]
        data = data[bound_x[0]:bound_x[1]+1,
                  bound_y[0]:bound_y[1]+1,
                  bound_z[0]:bound_z[1]+1]
        print(data.shape)
        data = VoxelGrid(DenseEncoding(data)).fill().encoding.dense[:]
        os.system(f'rm {out_path}')
        
        # rescale if size is smaller than min_size
        size = np.array(data.shape)
        if np.min(size) < min_size:
            size_valid = np.clip(size, min_size, None).astype(int)
            print(f"correct size: {size} -> {size_valid}")
            data = interpolate(torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0),
                size_valid.tolist(), mode='trilinear', align_corners=True)[0, 0].numpy() > 0.5

        voxel_list.append(data)

    os.system(f'rm {norm_path}')
    return voxel_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, help='source mesh path', required=True, default=None)
    parser.add_argument('--res', type=int, help='finest resolution', default=128)
    parser.add_argument('--factor', type=float, help='downsampling factor', default=0.75)
    parser.add_argument('--n_scales', type=int, help='number of scales', default=7)
    parser.add_argument('--min_size', type=int, help='minimum size', default=15)
    parser.add_argument('-o', '--output', type=str, help='output save path', default=None)
    args = parser.parse_args()

    # compute resolution hierarchy
    res_list = [args.res]
    for i in range(args.n_scales - 1):
        res_list.append(np.ceil(res_list[-1] * args.factor).astype(int))
    res_list = res_list[::-1]
    print(res_list)
    
    voxel_list = multiscale_voxelization_binvox(args.src, res_list, args.min_size)

    # save to .h5
    if args.output is None:
        save_path = args.src + f'_r{args.res}s{args.n_scales}.h5'
    else:
        save_path = args.output
        if not save_path.endswith('.h5'):
            save_path += ".h5"

    with h5py.File(save_path, 'w') as fp:
        fp.attrs['n_scales'] = len(voxel_list)
        for i in range(len(voxel_list)):
            vox = voxel_list[i].astype(bool)
            fp.create_dataset(f'scale{i}', data=vox, compression=9)
            print(vox.shape)
    print(save_path)
