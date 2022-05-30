import os
import h5py
import time
import torch
from config import get_config
from model import SSGmodel
from utils.file_utils import ensure_dir


def main():
    cfg = get_config('test')

    ssg_model = SSGmodel(cfg)

    # load from checkpoint
    n_scales = cfg.ckpt
    if cfg.ckpt is None:
        filename = sorted(os.listdir(cfg.model_dir))[-1]
        n_scales = int(filename.split('_')[0][-1])
    ssg_model.load_ckpt(n_scales)

    if cfg.mode == 'rand' or cfg.mode == 'rec':
        generate(cfg, ssg_model)
    elif cfg.mode == 'interp':
        interpolate(cfg, ssg_model)
    else:
        raise NotImplementedError


def generate(cfg, ssg_model: SSGmodel):
    """random generation or reconstruction"""
    out_name = f"{cfg.mode}_n{cfg.n_samples}"
    if cfg.bin:
        out_name += "_bin"
    if cfg.upsample > 1:
        out_name += f"_x{cfg.upsample}"
    out_name += f"_r{cfg.resize[0]}x{cfg.resize[1]}x{cfg.resize[2]}"    
    save_dir = os.path.join(cfg.exp_dir, out_name)
    ensure_dir(save_dir)

    for i in range(cfg.n_samples):
        since = time.time()
        with torch.no_grad():
            fake_ = ssg_model.generate(cfg.mode, resize_factor=cfg.resize, upsample=cfg.upsample)
        end = time.time()
        print(f"{i}. time:{end - since}.")
        fake_ = fake_.detach().cpu().numpy()[0, 0]
        if cfg.bin:
            fake_ = fake_ > 0.5
        
        save_path = os.path.join(save_dir, f"fake_{i:02d}.h5")
        _save_single(save_path, fake_, ssg_model.scale + 1)


def interpolate(cfg, ssg_model: SSGmodel):
    """interpolation and extrapolation. No resize, no upsample."""
    out_name = f"interp_n{cfg.n_samples}"
    if cfg.bin:
        out_name += "_bin"
    out_dir = os.path.join(cfg.exp_dir, out_name)
    ensure_dir(out_dir)

    # NOTE: hard-coded blending weights    
    alpha_list = [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]

    mode = 'rand'
    for k in range(cfg.n_samples):
        init_noise1 = ssg_model.draw_init_noise(mode)
        init_noise2 = ssg_model.draw_init_noise(mode)
        noises_list = ssg_model.draw_noises_list(mode)

        for i in range(len(alpha_list)):
            since = time.time()
            alpha = alpha_list[i]
            print("alpha=", alpha)
            init_noise = (1 - alpha) * init_noise1 + alpha * init_noise2
            with torch.no_grad():
                fake_ = ssg_model.netG(init_noise, ssg_model.real_sizes, noises_list, mode)
            end = time.time()
            print(f"{i}. time:{end - since}.")
            fake_ = fake_.detach().cpu().numpy()[0, 0]
            if cfg.bin:
                fake_ = fake_ > 0.5
            
            save_dir = os.path.join(out_dir, f'pair{k}')
            ensure_dir(save_dir)
            save_path = os.path.join(save_dir, f"fake_{i:02d}_alpha{alpha}.h5")
            _save_single(save_path, fake_, ssg_model.scale + 1)


def _save_single(save_path, shape, n_scales):
    fp = h5py.File(save_path, 'w')
    fp.attrs['n_scales'] = n_scales
    fp.create_dataset(f'scale{n_scales - 1}', data=shape, compression=9)
    fp.close()


if __name__ == '__main__':
    main()
