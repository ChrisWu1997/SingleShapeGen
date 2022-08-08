import os
import torch
from tqdm import tqdm
from config import Config
from model import get_model, SSGmodelBase
from utils.data_utils import load_data_fromH5, save_h5_single


def main():
    # create experiment config by parsing cmd-line arguments
    cfg = Config()

    # create model
    ssg_model = get_model(cfg)

    if cfg.is_train:
        # restore from checkpoint if provided
        if cfg.ckpt is not None:
            ssg_model.load_ckpt(cfg.ckpt)
            ssg_model.scale += 1

        # load training data
        real_data_list = load_data_fromH5(cfg.src_path)

        # start training
        ssg_model.train(real_data_list)
    else:
        # load from checkpoint
        n_scales = cfg.ckpt
        if cfg.ckpt is None: # otherwise load highest scale model by default
            filename = sorted(os.listdir(cfg.model_dir))[-1]
            n_scales = int(filename.split('_')[0][-1])
        ssg_model.load_ckpt(n_scales)

        if cfg.mode == 'rand' or cfg.mode == 'rec':
            generate(cfg, ssg_model)
        elif cfg.mode == 'interp':
            interpolate(cfg, ssg_model)
        else:
            raise NotImplementedError


def generate(cfg: Config, ssg_model: SSGmodelBase):
    """random generation or reconstruction"""
    out_name = f"{cfg.mode}_n{cfg.n_samples}"
    if cfg.bin:
        out_name += "_bin"
    if cfg.upsample > 1:
        out_name += f"_x{cfg.upsample}"
    out_name += f"_r{cfg.resize[0]}x{cfg.resize[1]}x{cfg.resize[2]}"    
    save_dir = os.path.join(cfg.exp_dir, out_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved at {save_dir}.")

    for i in tqdm(range(cfg.n_samples), desc="Generation"):
        with torch.no_grad():
            fake_ = ssg_model.generate(cfg.mode, resize_factor=cfg.resize, upsample=cfg.upsample)
        fake_ = fake_.detach().cpu().numpy()[0, 0]
        if cfg.bin:
            fake_ = fake_ > 0.5
        
        save_path = os.path.join(save_dir, f"fake_{i:02d}.h5")
        save_h5_single(save_path, fake_, ssg_model.scale + 1)


def interpolate(cfg: Config, ssg_model: SSGmodelBase):
    """interpolation and extrapolation. No resize, no upsample."""
    assert hasattr(ssg_model, "interpolation") # only implemented for SSGmodelTriplane
    out_name = f"interp_n{cfg.n_samples}"
    if cfg.bin:
        out_name += "_bin"
    out_dir = os.path.join(cfg.exp_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Results will be saved at {out_dir}.")

    # NOTE: hard-coded blending weights    
    alpha_list = [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    print("Alpha values:", alpha_list)

    for k in tqdm(range(cfg.n_samples), desc="Inter(extra)-polation"):
        save_dir = os.path.join(out_dir, f'pair{k}')
        os.makedirs(save_dir, exist_ok=True)

        fake_list = ssg_model.interpolation(alpha_list)
        for i, fake_ in enumerate(fake_list):
            fake_ = fake_.detach().cpu().numpy()[0, 0]
            if cfg.bin:
                fake_ = fake_ > 0.5

            save_path = os.path.join(save_dir, f"fake_{i:02d}_alpha{alpha_list[i]}.h5")
            save_h5_single(save_path, fake_, ssg_model.scale + 1)


if __name__ == '__main__':
    main()
