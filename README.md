# Learning to Generate 3D Shapes from a Single Example
TBA.

## Get started
Set up a conda environment with all dependencies:
```bash
conda env create -f environment.yml
conda activate ssg
```

## Run pretrained models
We provide pretrained models for all example shapes in the paper: [link TBA]() ([backup]()). Download and extract in `checkpoints` folder.

### Random generation
To randomly generate new shapes, run
```bash
python test.py --tag ssg_Acropolis_res128s6 -g 0 --n_samples 10 --mode rand
```
The results will be saved in `checkpoints/ssg_Acropolis_r128s6/rand_n10_bin_r1x1x1`.

Specify `--resize` to change the spatial dimensions. For example, `--resize 1.5 1.0 1.0` generates shapes whose size along x-axis are 1.5 times larger than original.

Specify `--upsample` to construct the output shape at a higher resolution. For example, `--upsample 2` results in 2 times higher resolution.

### Interpolation and extrapolation
For interpolation and extrapolation between two randomly generated samples, run
```bash
python test.py --tag ssg_Acropolis_res128s6 -g 0 --n_samples 5 --mode interp
```

### Visualize and export
To quickly visualize the generated shapes, run
```bash
python vis_export.py -s checkpoints/ssg_Acropolis_r128s6/rand_n10_bin_r1x1x1 -f mesh --smooth 3 --cleanup 
```
`--smooth` specifies Laplacian smoothing iterations. `--cleanup` keeps only the largest connected component.

Specify `--export obj` to export meshes in `obj` format.

## Data preparation
- data source
- binvox

## Training
- TBA

## Evaluation
- TBA

## Rendering
- TBA

## Acknowledgement
- TBA

## Citation
- TBA
