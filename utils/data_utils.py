import numpy as np
from scipy.ndimage.filters import gaussian_filter
import h5py
import trimesh
from trimesh.smoothing import filter_laplacian
from mcubes import marching_cubes
from skimage import morphology


def load_data_fromH5(path, smooth=True, only_finest=False):
    shape_list = []
    with h5py.File(path, 'r') as fp:
        n_scales = fp.attrs['n_scales']
        if only_finest:
            shape = fp[f'scale{n_scales - 1}'][:]
            return shape

        for i in range(n_scales):
            shape = fp[f'scale{i}'][:].astype(np.float)

            if smooth:
                shape = gaussian_filter(shape, sigma=0.5)
                shape = np.clip(shape, 0.0, 1.0)
            shape_list.append(shape)
    
    if shape_list[0].shape[0] > shape_list[1].shape[0]:
        shape_list = shape_list[::-1]

    return shape_list


def save_h5_single(save_path, shape, n_scales):
    fp = h5py.File(save_path, 'w')
    fp.attrs['n_scales'] = n_scales
    fp.create_dataset(f'scale{n_scales - 1}', data=shape, compression=9)
    fp.close()


def voxelGrid2mesh(shape, laplacian=0, color=None):
    shape = np.pad(shape, 1)
    vertices, faces = marching_cubes(shape, 0.5)
    mesh = trimesh.Trimesh(vertices, faces)
    if color is not None:
        mesh.visual.face_colors = np.array([color]).repeat(len(mesh.faces), 0)
    if laplacian > 0:
        mesh = filter_laplacian(mesh, iterations=laplacian)
    return mesh


def normalize_mesh(mesh: trimesh.Trimesh):
    verts = mesh.vertices
    min_corner = np.min(verts, axis=0)
    max_corner = np.max(verts, axis=0)
    box_size = np.abs(max_corner - min_corner)
    verts = verts - min_corner[np.newaxis]
    verts = verts * (1. / np.max(box_size))
    mesh.vertices = verts
    return mesh


def get_biggest_connected_compoent(voxels):
    labels, num = morphology.label(voxels, return_num=True)
    max_ind = 1
    max_count = 0
    for l in range(1, num):
        count = np.sum(labels == l)
        if count > max_count:
            max_count = count
            max_ind = l
    mask = labels == max_ind
    voxels[~mask] = 0
    return voxels
