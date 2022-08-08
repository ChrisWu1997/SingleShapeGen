import os
import argparse
from trimesh.voxel.encoding import DenseEncoding
from trimesh.voxel import VoxelGrid
from utils.data_utils import get_biggest_connected_compoent, load_data_fromH5, voxelGrid2mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, help="results folder or path", required=True)
    parser.add_argument('-f', '--format', type=str, default='mesh', choices=['mesh', 'voxel'], help="visualize format")
    parser.add_argument('--cleanup', action='store_true', help="only keep the biggest connected component")
    parser.add_argument('--smooth', type=int, default=0, help="laplacian smoothing iterations")
    parser.add_argument('--export', type=str, default=None, choices=['obj', 'glb'], help="export mesh format.")
    parser.add_argument('--no_vis', dest='vis', action='store_false', help='disable showing, only export')
    parser.set_defaults(vis=True)
    # TODO: add visuzalize all levels in the hierarchy
    args = parser.parse_args()

    color = [152, 199, 255, 255]
    if os.path.isdir(args.src):
        filepaths = [os.path.join(args.src, x) for x in sorted(os.listdir(args.src)) if x.endswith('.h5')]
    else:
        filepaths = [args.src]

    for path in filepaths:
        print(path)
        shape = load_data_fromH5(path, smooth=False, only_finest=True) > 0.5
        if args.cleanup:
            shape = shape.astype(int)
            shape = get_biggest_connected_compoent(shape)
        
        if args.format == 'voxel':
            # remove inner voxels for faster view
            voxel_grid = VoxelGrid(DenseEncoding(shape > 0.5)).hollow()
            voxel_grid.show()
        else:
            mesh = voxelGrid2mesh(shape, args.smooth, color)
            if args.vis:
                mesh.show()

            if args.export is not None:
                if os.path.isdir(args.src):
                    save_dir = args.src + f'_{args.export}'
                else:
                    save_dir = os.path.dirname(args.src) + f'_{args.export}'
                os.makedirs(save_dir, exist_ok=True)
                name = os.path.basename(path).split('.')[0]
                save_path = os.path.join(save_dir, name + f'.{args.export}')
                mesh.export(save_path)
