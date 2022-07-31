import os
import random
import numpy as np
import argparse
import torch
from tqdm import tqdm
import sys
sys.path.append('..')
from utils.data_utils import load_data_fromH5


def pairwise_IoU_dist(data_list: list):
    """average pairwise 1-IoU for a list of 3D shape volume"""
    avgv = []
    for i in tqdm(range(len(data_list)), desc='Div'):
        data_i = data_list[i]
        intersect = torch.logical_and(data_i, data_list).sum(dim=(1, 2, 3))
        union = torch.logical_or(data_i, data_list).sum(dim=(1, 2, 3))
        iou_dist = 1.0 - intersect / union
        mask = torch.ones_like(iou_dist, dtype=torch.bool)
        mask[i] = False
        iou_dist = iou_dist[mask]
        avgv.append(torch.mean(iou_dist).item())
    avgv = np.mean(avgv)
    return avgv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, required=True, help='generated data folder')
    parser.add_argument('-o', '--output', type=str, default=None, help='result save path')
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, help="which gpu to use. -1 for CPU.")
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    device = torch.device(f"cuda:{args.gpu_ids}" if args.gpu_ids >= 0 else "cpu")

    random.seed(1234)

    gen_data_list = []
    filenames = sorted([x for x in os.listdir(args.src) if x.endswith('.h5')])
    for name in filenames:
        path = os.path.join(args.src, name)
        gen_data = load_data_fromH5(path, smooth=False, only_finest=True)
        gen_data = torch.from_numpy(gen_data > 0.5).to(device)
        
        gen_data_list.append(gen_data)

    gen_data_list = torch.stack(gen_data_list, dim=0)
    div = pairwise_IoU_dist(gen_data_list).round(6)
    
    eval_results = {'Div': div}

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
