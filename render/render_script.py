import os
import argparse
from PIL import Image
import sys
sys.path.append('..')
from utils.data_utils import load_data_fromH5, normalize_mesh, voxelGrid2mesh
from blender_utils import BLENDER_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, required=True, help="path to data, i.e. h5 file(s)")
    parser.add_argument('-o', '--output', type=str, default=None, help="output path")
    parser.add_argument('-c', '--config', type=str, default='default', help="saved object/lighting configuration")
    parser.add_argument('--mesh_color', type=str, default='blue', choices=['red', 'blue', 'green'])
    # parser.add_argument('--box', action='store_true', help="use box-like mesh for voxel")
    parser.add_argument('--cleanup', action='store_true', help="only keep the biggest connected component")
    parser.add_argument('--smooth', type=int, default=0, help="laplacian smoothing iterations")
    args = parser.parse_args()

    if os.path.isdir(args.src):
        filenames = [os.path.join(args.src, x) for x in sorted(os.listdir(args.src)) if x.endswith('h5')]
        out_dir = args.output if args.output is not None else args.src + '_render'
    else:
        assert args.src.endswith('h5')
        filenames = [args.src]
        out_dir = args.output if args.output is not None else os.path.dirname(args.src) + '_render'
    if args.smooth > 0:
        out_dir += f'_lap'
    if args.cleanup > 0:
        out_dir += f'_clean'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for path in filenames:
        print(path)

        shape = load_data_fromH5(path, smooth=False, only_finest=True)
        mesh = voxelGrid2mesh(shape, args.smooth)
        mesh = normalize_mesh(mesh)

        name = os.path.splitext(os.path.basename(path))[0]
        mesh_path = os.path.join(out_dir, name + '.obj')
        print('->', mesh_path)
        mesh.export(mesh_path)

        # render mesh using Blender
        cmd = f'{BLENDER_PATH} --background --python blender_render.py -- -s {mesh_path} -c {args.config} --mesh_color {args.mesh_color}'
        os.system(cmd)

        # detele mesh file
        os.remove(mesh_path)

        # crop image
        image_path = os.path.splitext(mesh_path)[0] + '.png'
        print(image_path)
        im = Image.open(image_path)
        print("crop image:", im.getbbox())
        im2 = im.crop(im.getbbox())
        im2.save(image_path)
