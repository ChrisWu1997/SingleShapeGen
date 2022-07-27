# Learning to Generate 3D Shapes from a Single Example
TBA.

## Get started
Set up a conda environment with all dependencies:
```bash
conda env create -f environment.yml
conda activate ssg
```
TBA: A Colab?

## Run pretrained models
We provide pretrained models for all example shapes in the paper: [link TBA]() ([backup]()). Download and extract in `checkpoints` folder.

### Random generation
To randomly generate new shapes, run
```bash
python test.py --tag ssg_Acropolis_res256s8 -g 0 --n_samples 10 --mode rand
```
The results will be saved in `checkpoints/ssg_Acropolis_r256s8/rand_n10_bin_r1x1x1`.

Specify `--resize` to change the spatial dimensions. For example, `--resize 1.5 1.0 1.0` generates shapes whose size along x-axis are 1.5 times larger than original.

Specify `--upsample` to construct the output shape at a higher resolution. For example, `--upsample 2` results in 2 times higher resolution.

### Interpolation and extrapolation
For interpolation and extrapolation between two randomly generated samples, run
```bash
python test.py --tag ssg_Acropolis_r256s8 -g 0 --n_samples 5 --mode interp
```

### Visualize and export
To quickly visualize the generated shapes, run
```bash
python vis_export.py -s checkpoints/ssg_Acropolis_r256s8/rand_n10_bin_r1x1x1 -f mesh --smooth 3 --cleanup
```
`--smooth` specifies Laplacian smoothing iterations. `--cleanup` keeps only the largest connected component.

Specify `--export obj` to export meshes in `obj` format.

## Data preparation
We list the sources for all example shapes that we used: [data/README.md](data/README.md). Most of them are free and you can download accordingly.

To construct the training data (voxel pyramid) from a mesh, we rely on [binvox](https://www.patrickmin.com/binvox/).
After downloading it, make sure you change [BINVOX_PATH]() in `voxelization/voxelize.py` to your path to excetuable binvox.
Then run our script
```bash
cd voxelization
python voxelize.py -s {path-to-your-mesh-file} --res 128 --n_scales 6 -o {save-path.h5}
# --res: finest voxel resolution. --n_scales: number of scales.
```
The processed data will be saved in `.h5` format.

TBA: how to provide preprocessed data?

## Training
To train on the processed h5 data, run
```bash
python train.py --tag {your-experiment-tag} -s {path-to-processed-h5-data} -g {gpu-id}
```
By default, the log and model will be saved in `checkpoints/{your-experiment-tag}`.

## Evaluation
We provide code for evaluation metrics LP-IoU, LP-F-score, SSFID and Div.
SSFID relies on a pretrained 3D shape classifier. Please download it from [here](https://drive.google.com/file/d/1HjnDudrXsNY4CYhIGhH4Q0r3-NBnBaiC/view?usp=sharing) and put `Clsshapenet_128.pth` under `evaluation` folder.

To perform evaluation, we first randomly generate 100 shapes, e.g.,
```bash
python test.py --tag ssg_Acropolis_res128s6 -g 0 --n_samples 100 --mode rand
```

Then run the evalution script to compute all metrics, e.g.,
```bash
cd evaluation
# ./eval.sh {generated-shapes-folder} {reference-shape} {gpu-ids}
./eval.sh ../checkpoints/ssg_Acropolis_r128s6/rand_n100_bin_r1x1x1 ../data/Acropolis_r128s6.h5 0
```
See `evaluation` folder for evalution scripts for each individual metric.

## Rendering
We provide code and configurations for rendering figures in our paper.
We rely on [Blender](https://www.blender.org) and [BlenderToolbox](https://github.com/HTDerekLiu/BlenderToolbox).
To use our rendering script, make sure have them installed and change the corresponding paths [BLENDER_PATH]() and [BLENDERTOOLBOX_PATH]().
Then run
```bash
cd render
python render_script.py -s {path-to-generated-shapes-folder} -c {render-config-name} --smooth 3 --cleanup
```
See `render/render_configs.json` for saved rendering configs.

## Acknowledgement
We develop some part of this repo based on code from [ConSinGAN](https://github.com/tohinz/ConSinGAN), [DECOR-GAN](https://github.com/czq142857/DECOR-GAN) and [BlenderToolbox](https://github.com/HTDerekLiu/BlenderToolbox). We would like to thank their authors.

## Citation
- TBA

## License
- TBA
