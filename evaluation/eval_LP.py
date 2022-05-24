import os
import random
from collections import OrderedDict
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
sys.path.append('..')
from utils.data_utils import load_data_fromH5


def extract_valid_patches_unfold(voxels, patch_size, stride=None):
    overlap = patch_size // 2 if stride is None else stride

    p = patch_size // 2
    voxels = F.pad(voxels, [p, p, p, p, p, p])
    patches = voxels.unfold(0, patch_size, overlap).unfold(1, patch_size, overlap).unfold(2, patch_size, overlap) 
    patches = patches.contiguous().view(-1, patch_size, patch_size, patch_size) # (k, ps, ps, ps)
    
    # valid patch criterion
    # center region (l^3) has at least one occupied and one unoccupied voxel
    idx = patch_size // 2 - 1
    l = 2 if patch_size % 2 == 0 else 3
    centers = patches[:, idx:idx+l, idx:idx+l, idx:idx+l] # (k, l, l, l)
    mask_occ = torch.sum(centers.int(), dim=(1, 2, 3)) > 0 # (k,)
    mask_unocc = torch.sum(centers.int(), dim=(1, 2, 3)) < l * l * l # (k,)
    mask = torch.logical_and(mask_occ, mask_unocc)

    patches = patches[mask]
    return patches


def eval_LP_IoU(gen_patches, ref_patches, threshold=0.95):
    dists = []
    for i in range(gen_patches.shape[0]):
        intersect = torch.logical_and(ref_patches, gen_patches[i:i+1]).sum(dim=(1, 2, 3))
        union = torch.logical_or(ref_patches, gen_patches[i:i+1]).sum(dim=(1, 2, 3))
        max_iou = torch.max(intersect / union)
        dists.append(max_iou)
    dists = torch.stack(dists)
    avg_iou = torch.mean(dists).item()
    percent = torch.sum((dists > threshold).int()).item() * 1.0 / len(dists)
    return avg_iou, percent


def eval_LP_Fscore(gen_patches, ref_patches, threshold=0.95):
    dists = []
    for i in range(gen_patches.shape[0]):
        true_positives = torch.logical_and(ref_patches, gen_patches[i:i+1]).sum(dim=(1, 2, 3))
        precision = true_positives / gen_patches[i:i+1].sum()
        recall = true_positives / ref_patches.sum(dim=(1, 2, 3))
        Fscores = 2 * precision * recall / (precision + recall + 1e-8)
        Fscore = torch.max(Fscores)
        dists.append(Fscore)
    dists = torch.stack(dists)
    avg_fscore = torch.mean(dists).item()
    percent = torch.sum((dists > threshold).int()).item() * 1.0 / len(dists)
    return avg_fscore, percent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, required=True, help='generated data folder')
    parser.add_argument('-r', '--ref', type=str, required=True, help='reference data path')
    parser.add_argument('--patch_size', type=int, default=11, help='patch size')
    parser.add_argument('--stride', type=int, default=None, help='patch stride. By default, half of patch size.')
    parser.add_argument('--patch_num', type=int, default=1000, help='max number of patches sampled from generated shapes.')
    parser.add_argument('-o', '--output', type=str, default=None, help='result save path')
    parser.add_argument('-g', '--gpu_ids', type=str, default=0, help="gpu to use, e.g. 0  0,1,2. CPU not supported.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

    random.seed(1234)

    # load real
    ref_data = load_data_fromH5(args.ref, smooth=False, only_finest=True)
    ref_data = torch.from_numpy(ref_data > 0.5).cuda()
    
    print('ref shape size:', ref_data.shape)
    ref_patches = extract_valid_patches_unfold(ref_data, args.patch_size, args.stride)
    ref_patches = ref_patches
    # print('ref_patches:', ref_patches.shape)

    # LP
    result_lp_iou_percent = []
    result_lp_fscore_percent = []

    filenames = sorted([x for x in os.listdir(args.src) if x.endswith('.h5')])
    for name in tqdm(filenames):
        path = os.path.join(args.src, name)
        gen_data = load_data_fromH5(path, smooth=False, only_finest=True)
        gen_data = torch.from_numpy(gen_data > 0.5).cuda()

        gen_patches = extract_valid_patches_unfold(gen_data, args.patch_size, args.stride)
        indices = list(range(gen_patches.shape[0]))
        random.shuffle(indices)
        indices = indices[:args.patch_num]
        gen_patches = gen_patches[indices]

        lp_iou_avg, lp_iou_percent = eval_LP_IoU(gen_patches, ref_patches)
        lp_fscore_avg, lp_fscore_percent = eval_LP_Fscore(gen_patches, ref_patches)

        result_lp_iou_percent.append(lp_iou_percent)
        result_lp_fscore_percent.append(lp_fscore_percent)

    result_lp_iou_percent = np.mean(result_lp_iou_percent).round(6)
    result_lp_fscore_percent = np.mean(result_lp_fscore_percent).round(6)
    
    eval_results = OrderedDict({'LP-IOU': result_lp_iou_percent,
                                'LP-F-score': result_lp_fscore_percent})

    if args.output is None:
        save_path = args.src + f'_eval_LP_p{args.patch_size}s{args.stride}n{args.patch_num}.txt'
    else:
        save_path = args.output

    fp = open(save_path, 'w')
    for k, v in eval_results.items():
        print(f"{k}: {v}")
        print(f"{k}: {v}", file=fp)


if __name__ == '__main__':
    main()
