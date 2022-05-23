import os
import time
from collections import OrderedDict
import numpy as np
import json
import torch
from tqdm import tqdm
from utils import ensure_dir
from dataset import generate_3Ddata_multiscale
from common import get_config
from agent import get_agent


def binarize(arr):
    return torch.where(arr > 0.5, torch.ones_like(arr), torch.zeros_like(arr)).bool()


def iou(arr1, arr2):
    intersect = torch.sum(torch.logical_and(arr1, arr2).int()).item() * 1.0
    union = torch.sum(torch.logical_or(arr1, arr2).int()).item() * 1.0
    return intersect / union


# def extract_nonempty_patches(voxel, size, threshold=0.05):
#     # since = time.time()
#     overlap = size // 2
#     patch_list = []
#     for i in range(0, voxel.shape[0] - size, overlap):
#         for j in range(0, voxel.shape[1] - size, overlap):
#             for k in range(0, voxel.shape[2] - size, overlap):
#                 patch = voxel[i:i + size, j:j + size, k:k + size]
#                 occ = torch.sum(patch) * 1.0 / (size ** 3)
#                 if threshold < occ < 1 - threshold:
#                     patch_list.append(patch)
#     if len(patch_list) == 0:
#         return None
#     patch_list = torch.stack(patch_list)
#     # now = time.time()
#     # print("patch time:", now - since)
#     return patch_list


def extract_nonempty_patches_unfold(voxel, size, threshold=0.05, stride=None):
    overlap = size // 2 if stride is None else stride

    patches = voxel.unfold(0, size, overlap).unfold(1, size, overlap).unfold(2, size, overlap) # (M, N, L, ps, ps, ps)
    patches = patches.contiguous().view(-1, size, size, size)
    
    # valid patch criterion
    occ = torch.sum(patches.int(), dim=(1, 2, 3)) / (size ** 3)
    mask = torch.logical_and(occ > threshold, occ < 1 - threshold)
    patches = patches[mask]
    return patches


def max_patch_IoU(fake_patches, real_patches, threshold=0.95):
    dists = []
    for i in range(fake_patches.shape[0]):
        intersect = torch.logical_and(real_patches, fake_patches[i:i+1]).sum(dim=(1, 2, 3))
        union = torch.logical_or(real_patches, fake_patches[i:i+1]).sum(dim=(1, 2, 3))
        max_iou = torch.max(intersect / union)
        dists.append(max_iou)
    dists = torch.stack(dists)
    avg_iou = torch.mean(dists).item()
    percent = torch.sum((dists > threshold).int()).item() * 1.0 / len(dists)
    return avg_iou, percent


def mutual_IoU(fake_list):
    maxv, avgv = [], []
    for i in range(len(fake_list)):
        fake = fake_list[i]
        intersect = torch.logical_and(fake, fake_list).sum(dim=(1, 2, 3))
        union = torch.logical_or(fake, fake_list).sum(dim=(1, 2, 3))
        ious = intersect / union
        mask = torch.ones_like(ious, dtype=torch.bool)
        mask[i] = False
        ious = ious[mask]
        maxv.append(torch.max(ious).item())
        avgv.append(torch.mean(ious).item())
    maxv = np.mean(maxv)
    avgv = np.mean(avgv)
    return maxv, avgv


def main():
    # create experiment config containing all hyperparameters
    config = get_config('test')

    # create network and training agent
    tr_agent = get_agent(config)

    # load from checkpoint if provided
    tr_agent.load_ckpt(config.n_scales)

    real_data_list = generate_3Ddata_multiscale(config)
    tr_agent.set_real_data(real_data_list)

    save_dir = os.path.join(config.exp_dir, "outputs")
    ensure_dir(save_dir)

    eval_results = OrderedDict()

    PATCH_SIZE = 12
    N = 100

    # real 
    real_bin = real_data_list[-1].detach().squeeze(0).squeeze(0)
    real_bin = binarize(real_bin)
    real_patches = extract_nonempty_patches_unfold(real_bin, PATCH_SIZE)
    # real_patches = extract_nonempty_patches(real_bin, PATCH_SIZE)
    real_patches_2x = extract_nonempty_patches_unfold(real_bin, PATCH_SIZE * 2)

    # generation
    fake_list = []
    shape_iou_list = []
    patch_iou_list = []
    patch_percent_list = []
    patch_iou_2x_list = []
    patch_percent_2x_list = []
    for i in tqdm(range(N)):
        fake_ = tr_agent.generate('rand')
        fake_ = fake_.detach().squeeze(0).squeeze(0)
        fake_ = binarize(fake_)

        fake_iou = iou(fake_, real_bin)
        shape_iou_list.append(fake_iou)

        fake_patches = extract_nonempty_patches_unfold(fake_, PATCH_SIZE)
        # fake_patches = extract_nonempty_patches(fake_, PATCH_SIZE)
        patch_iou, patch_percent = max_patch_IoU(fake_patches, real_patches)
        fake_patches_2x = extract_nonempty_patches_unfold(fake_, PATCH_SIZE * 2)
        patch_iou2x, patch_percent2x = max_patch_IoU(fake_patches_2x, real_patches_2x)

        fake_ = fake_.cpu()
        fake_list.append(fake_)
        patch_iou_list.append(patch_iou)
        patch_percent_list.append(patch_percent)
        patch_iou_2x_list.append(patch_iou2x)
        patch_percent_2x_list.append(patch_percent2x)
    
    iou_avg = np.mean(shape_iou_list)
    patch_iou_avg = np.mean(patch_iou_list)
    patch_percent_avg = np.mean(patch_percent_list)
    patch_iou_2x_avg = np.mean(patch_iou_2x_list)
    patch_percent_2x_avg = np.mean(patch_percent_2x_list)
    eval_results.update({'fake_iou': iou_avg, 'patch_iou': patch_iou_avg, 'patch_percent': patch_percent_avg,
                         'patch_iou_2x': patch_iou_2x_avg, 'patch_percent_2x': patch_percent_2x_avg})
    
    # diversity measures
    fake_list = torch.stack(fake_list, dim=0)
    max_mIoU, avg_mIoU = mutual_IoU(fake_list)

    fake_mean = torch.mean(fake_list.float(), dim=0)
    fake_std = torch.std(fake_list.float(), dim=0)
    mask = fake_mean > 1. / N # filter empty region
    nonempty_ratio = torch.mean(mask.float()).item()
    fake_std = fake_std[mask]
    fake_std_avg = torch.mean(fake_std).item()
    fake_std_med = torch.median(fake_std).item()
    eval_results.update({'fake_nonempty_ratio': nonempty_ratio, 'voxel_std_avg': fake_std_avg, 'voxel_std_med': fake_std_med,
                         'muIoU_max': max_mIoU, 'muIoU_avg': avg_mIoU})

    save_path = os.path.join(config.exp_dir, "eval_results.txt")
    fp = open(save_path, 'w')
    for k, v in eval_results.items():
        eval_results[k] = round(v, 6)
        print(f"{k}: {v}")
        print(f"{k}: {v}", file=fp)
    
    print('#####' * 6, file=fp)
    
    for k, v in eval_results.items():
        print(v, file=fp)
    fp.close()


if __name__ == '__main__':
    main()
