import os
import random
import numpy as np
import torch
from collections import OrderedDict
from scipy import linalg
import argparse
from tqdm import tqdm
from classifier3D import classifier
import sys
sys.path.append('..')
from utils.data_utils import load_data_fromH5


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(voxel, model, model_out_layer=2):
    """Calculation of the statistics used by the FID.
    Returns:
    -- mu    : The mean over samples of the activations of the inception model.
    -- sigma : The covariance matrix of the activations of the inception model.
    """
    # act = get_activations(files, model, batch_size, dims, cuda, verbose)
    with torch.no_grad():
        act = model(voxel.unsqueeze(0).unsqueeze(0), out_layer=model_out_layer)
    act = act.permute(0, 2, 3, 4, 1).view(-1, act.shape[1]).detach().cpu().numpy() # (D*H*W, C)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, required=True, help='generated data folder')
    parser.add_argument('-r', '--ref', type=str, required=True, help='reference data path')
    parser.add_argument('--model_out_layer', type=int, default=2, help='use the output from which layer')
    parser.add_argument('-o', '--output', type=str, default=None, help='result save path')
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, help="which gpu to use. -1 for CPU.")
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    device = torch.device(f"cuda:{args.gpu_ids}" if args.gpu_ids >= 0 else "cpu")

    random.seed(1234)

    # load model
    model = classifier()
    voxel_size = 128
    weights_path = 'Clsshapenet_'+str(voxel_size)+'.pth'
    if not os.path.exists(weights_path):
        raise RuntimeError(f"'{weights_path}' not exists. Please download it first.")
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()

    # load ref
    ref_data = load_data_fromH5(args.ref, smooth=False, only_finest=True)
    ref_data = torch.from_numpy(ref_data).float().to(device)

    mu_r, sigma_r = calculate_activation_statistics(ref_data, model, args.model_out_layer)

    filenames = sorted([x for x in os.listdir(args.src) if x.endswith('.h5')])
    ssfid_values = []
    for name in tqdm(filenames, desc="SSFID"):
        path = os.path.join(args.src, name)
        gen_data = load_data_fromH5(path, smooth=False, only_finest=True)
        gen_data = torch.from_numpy(gen_data).float().to(device)
        if gen_data.shape != ref_data.shape:
            raise RuntimeError('Generated shape and reference shape shall have equal size.')
        
        mu_f, sigma_f = calculate_activation_statistics(gen_data, model, args.model_out_layer)
        
        ssfid = calculate_frechet_distance(mu_r, sigma_r, mu_f, sigma_f)
        ssfid_values.append(ssfid)
    
    ssfid_avg = np.mean(ssfid_values).round(6)
    ssfid_std = np.std(ssfid_values).round(6)

    eval_results = OrderedDict({'SSFID_avg': ssfid_avg,
                                'SSFID_std': ssfid_std,})
    
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
