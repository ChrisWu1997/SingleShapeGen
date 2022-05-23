import os
import h5py
import time
from config import get_config
from model import SSGmodel
from utils.file_utils import ensure_dir


def main():
    # create experiment cfg containing all hyperparameters
    cfg = get_config('test')

    # create network and training agent
    mymodel = SSGmodel(cfg)

    # load from checkpoint if provided
    n_scales = len(os.listdir(cfg.model_dir))
    mymodel.load_ckpt(n_scales)

    out_name = f"out_{cfg.test_mode}_n{cfg.n_samples}"
    if cfg.bin:
        out_name += "_bin"
    out_name += f"_r{cfg.resize[0]}x{cfg.resize[1]}x{cfg.resize[2]}"    
    save_dir = os.path.join(cfg.exp_dir, out_name)
    ensure_dir(save_dir)

    end_scale = mymodel.scale
    for i in range(cfg.n_samples):
        since = time.time()
        fake_ = mymodel.generate(cfg.test_mode, end_scale - 1, resize_factor=cfg.resize)
        fake_list = [fake_]
        end = time.time()
        print(f"{i}. time:{end - since}.")
        
        save_path = os.path.join(save_dir, f"fake_{i:02d}.h5")
        fp = h5py.File(save_path, 'w')
        fp.attrs['n_scales'] = end_scale
        for j, fake_ in enumerate(fake_list):
            fake_ = fake_.detach().cpu().numpy()[0, 0]
            if cfg.bin:
                fake_ = fake_ > 0.5
            fp.create_dataset(f'scale{end_scale - 1 - j}', data=fake_, compression=9)
        fp.close()


if __name__ == '__main__':
    main()
