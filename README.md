# Learning to Generate 3D Shapes from a Single Example

![teaser](https://user-images.githubusercontent.com/32172140/182961115-c4d5aa26-b9c5-4f44-afda-28d0f785c6d3.jpg)

Official implementation for the paper:
> **[Learning to Generate 3D Shapes from a Single Example](http://www.cs.columbia.edu/cg/SingleShapeGen/)**  
> [Rundi Wu](https://www.cs.columbia.edu/~rundi/), [Changxi Zheng](http://www.cs.columbia.edu/~cxz/)  
> Columbia University  
> SIGGRAPH Asia 2022 (Journal Track)


## Installation
Prerequisites:
- python 3.9+
- An Nvidia GPU

Install dependencies with conda:
```bash
conda env create -f environment.yml
conda activate ssg
```
or install dependencies with pip:
```bash
pip install -r requirement.txt
# NOTE: check https://pytorch.org/ for pytorch installation command for your CUDA version
```


## Pretrained models
We provide pretrained models for all shapes that are used in our paper. Download all of them (about 1G):
```bash
bash download_models.sh
```
or download each model individually, e.g.,
```bash
bash download_models.sh ssg_Acropolis_r256s8
```
[Backup Google Drive link](https://drive.google.com/drive/folders/1kgiKxdsRnFryHQKMX5NbKja155NZeiMD?usp=sharing).


## Quick start: a GUI demo
We provide a simple GUI demo (based on [Open3D](https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/vis_gui.py)) that allows quick shape generation with a trained model. For example, run
```bash
python gui_demo.py checkpoints/ssg_Acropolis_r256s8
```
![ssg_demo](https://user-images.githubusercontent.com/32172140/182960381-f3e38725-5565-4baf-972f-f4dd085c65c0.gif)

(Recorded on a Ubuntu 20.04 with an NVIDIA 3090 GPU. Also tested on a Window 11 with an NVIDIA 2070 GPU.) 

## Inference

### Random generation
To randomly generate new shapes, run
```bash
python main.py test --tag ssg_Acropolis_r256s8 -g 0 --n_samples 10 --mode rand
```
The generated shapes will be saved in `.h5` format, compatible with the training data.

Specify `--resize` to change the spatial dimensions. For example, `--resize 1.5 1.0 1.0` generates shapes whose size along x-axis are 1.5 times larger than original.

Specify `--upsample` to construct the output shape at a higher resolution. For example, `--upsample 2` gives in 2 times higher resolution.


### Interpolation and extrapolation
For interpolation and extrapolation between two randomly generated samples, run
```bash
python main.py test --tag ssg_Acropolis_r256s8 -g 0 --n_samples 5 --mode interp
```


### Visualize and export
To quickly visualize the generated shapes (of `.h5` format), run
```bash
python vis_export.py -s checkpoints/ssg_Acropolis_r256s8/rand_n10_bin_r1x1x1 -f mesh --smooth 3 --cleanup
```
`--smooth` specifies Laplacian smoothing iterations. `--cleanup` keeps only the largest connected component.

Specify `--export obj` to export meshes in `obj` format.


## Training data preparation
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

TBA: release preprocessed data?


## Training
To train on the processed h5 data, run
```bash
python main.py train --tag {your-experiment-tag} -s {path-to-processed-h5-data} -g {gpu-id}
```
By default, the log and model will be saved in `checkpoints/{your-experiment-tag}`.


## Evaluation
We provide code for evaluation metrics LP-IoU, LP-F-score, SSFID and Div.
SSFID relies on a pretrained 3D shape classifier. Please download it from [here](https://drive.google.com/file/d/1HjnDudrXsNY4CYhIGhH4Q0r3-NBnBaiC/view?usp=sharing) and put `Clsshapenet_128.pth` under `evaluation` folder.

To perform evaluation, we first randomly generate 100 shapes, e.g.,
```bash
python main.py test --tag ssg_Acropolis_r128s6 -g 0 --n_samples 100 --mode rand
```

Then run the evalution script to compute all metrics, e.g.,
```bash
cd evaluation
# ./eval.sh {generated-shapes-folder} {reference-shape} {gpu-ids}
./eval.sh ../checkpoints/ssg_Acropolis_r128s6/rand_n100_bin_r1x1x1 ../data/Acropolis_r128s6.h5 0
```
See `evaluation` folder for evalution scripts for each individual metric.


## SinGAN-3D baseline
We also provide a baseline, SinGAN-3D, as described in our paper. To use it, simply specify `--G_struct conv3d` when training the model. Pretrained models are also provided (begin with "singan3d").


## Rendering
We provide code and configurations for rendering figures in our paper.
We rely on [Blender](https://www.blender.org) and [BlenderToolbox](https://github.com/HTDerekLiu/BlenderToolbox).
To use our rendering script, make sure have them installed and change the corresponding paths ([BLENDER_PATH]() and [BLENDERTOOLBOX_PATH]() in `render/blender_utils.py`).
Then run
```bash
cd render
python render_script.py -s {path-to-generated-shapes-folder} -c {render-config-name} --smooth 3 --cleanup
```
See `render/render_configs.json` for saved rendering configs.


## Acknowledgement
We develop some part of this repo based on code from [SinGAN](https://github.com/tamarott/SinGAN), [ConSinGAN](https://github.com/tohinz/ConSinGAN), [DECOR-GAN](https://github.com/czq142857/DECOR-GAN) and [BlenderToolbox](https://github.com/HTDerekLiu/BlenderToolbox). We would like to thank their authors.

## Citation
```
@article{wu2022learning,
    title={Learning to Generate 3D Shapes from a Single Example},
    author={Wu, Rundi and Zheng, Changxi},
    journal={ACM Transactions on Graphics (TOG)},
    volume={41},
    number={6},
    articleno={224},
    numpages={19},
    year={2022},
    publisher={ACM New York, NY, USA}
}
```
