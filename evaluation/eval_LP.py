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


def extract_valid_patches_unfold(voxels: torch.Tensor, patch_size: int, stride=None):
    """extract near-surface patches of a 3D shape using torch.unfold

    Args:
        voxels (torch.Tensor): a 3D shape volume of size (H, W, D)
        patch_size (int): patch size
        stride (int, optional): stride for overlapping. Defaults to None. If None, set as half patch size.

    Returns:
        patches: size (N, patch_size, patch_size, patch_size)
    """
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


def eval_LP_IoU(gen_patches: torch.Tensor, ref_patches: torch.Tensor, threshold=0.95):
    """compute LP-IoU over two set of patches.

    Args:
        gen_patches (torch.Tensor): patches from generated shape
        ref_patches (torch.Tensor): patches from reference shape
        threshold (float, optional): IoU threshold. Defaults to 0.95.

    Returns:
        average max IoU, LP-IoU
    """
    values = []
    for i in range(gen_patches.shape[0]):
        intersect = torch.logical_and(ref_patches, gen_patches[i:i+1]).sum(dim=(1, 2, 3))
        union = torch.logical_or(ref_patches, gen_patches[i:i+1]).sum(dim=(1, 2, 3))
        max_iou = torch.max(intersect / union)
        values.append(max_iou)
    values = torch.stack(values)
    avg_iou = torch.mean(values).item()
    percent = torch.sum((values > threshold).int()).item() * 1.0 / len(values)
    return avg_iou, percent


def eval_LP_Fscore(gen_patches: torch.Tensor, ref_patches: torch.Tensor, threshold=0.95):
    """compute LP-F-score over two set of patches.

    Args:
        gen_patches (torch.Tensor): patches from generated shape
        ref_patches (torch.Tensor): patches from reference shape
        threshold (float, optional): F-score threshold. Defaults to 0.95.

    Returns:
        average max F-score, LP-F-score
    """
    values = []
    for i in range(gen_patches.shape[0]):
        true_positives = torch.logical_and(ref_patches, gen_patches[i:i+1]).sum(dim=(1, 2, 3))
        precision = true_positives / gen_patches[i:i+1].sum()
        recall = true_positives / ref_patches.sum(dim=(1, 2, 3))
        Fscores = 2 * precision * recall / (precision + recall + 1e-8)
        Fscore = torch.max(Fscores)
        values.append(Fscore)
    values = torch.stack(values)
    avg_fscore = torch.mean(values).item()
    percent = torch.sum((values > threshold).int()).item() * 1.0 / len(values)
    return avg_fscore, percent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, required=True, help='generated data folder')
    parser.add_argument('-r', '--ref', type=str, required=True, help='reference data path')
    parser.add_argument('--patch_size', type=int, default=11, help='patch size')
    parser.add_argument('--stride', type=int, default=None, help='patch stride. By default, half of patch size.')
    parser.add_argument('--patch_num', type=int, default=1000, help='max number of patches sampled from generated shapes.')
    parser.add_argument('-o', '--output', type=str, default=None, help='result save path')
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, help="which gpu to use. -1 for CPU.")
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    device = torch.device(f"cuda:{args.gpu_ids}" if args.gpu_ids >= 0 else "cpu")
    args.stride = args.patch_size // 2 if args.stride is None else args.stride

    random.seed(1234)

    # load real
    ref_data = load_data_fromH5(args.ref, smooth=False, only_finest=True)
    ref_data = torch.from_numpy(ref_data > 0.5).to(device)
    
    ref_patches = extract_valid_patches_unfold(ref_data, args.patch_size, args.stride)
    ref_patches = ref_patches

    # LP
    result_lp_iou_percent = []
    result_lp_fscore_percent = []

    filenames = sorted([x for x in os.listdir(args.src) if x.endswith('.h5')])
    for name in tqdm(filenames, desc="LP-IOU/F-score"):
        path = os.path.join(args.src, name)
        gen_data = load_data_fromH5(path, smooth=False, only_finest=True)
        gen_data = torch.from_numpy(gen_data > 0.5).to(device)

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
        save_path = args.src + f'_eval.txt'
    else:
        save_path = args.output

    fp = open(save_path, 'a')
    for k, v in eval_results.items():
        print(f"{k}: {v}")
        print(f"{k}: {v}", file=fp)


if __name__ == '__main__':
    main()
