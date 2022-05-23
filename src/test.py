import os
import h5py
import time
from utils import ensure_dir
from dataset import generate_3Ddata_multiscale
from common import get_config
from model import get_agent


def main():
    # create experiment config containing all hyperparameters
    config = get_config('test')

    # create network and training agent
    tr_agent = get_agent(config)

    # load from checkpoint if provided
    tr_agent.load_ckpt(config.n_scales)

    real_data_list = generate_3Ddata_multiscale(config)
    tr_agent.set_real_data(real_data_list)

    out_name = f"out_{config.test_mode}_n{config.n_samples}"
    if config.bin:
        out_name += "_bin"
    if config.seq:
        out_name += "_seq"
    out_name += f"_r{config.resize[0]}x{config.resize[1]}x{config.resize[2]}"    
    save_dir = os.path.join(config.exp_dir, out_name)
    ensure_dir(save_dir)

    for i in range(config.n_samples):
        since = time.time()
        if config.seq:
            fake_list = tr_agent.generate(config.test_mode, config.n_scales - 1, return_each=True)
            fake_list = fake_list[::-1]
        else:
            fake_ = tr_agent.generate(config.test_mode, config.n_scales - 1, resize_factor=config.resize)
            fake_list = [fake_]
        end = time.time()
        print(f"{i}. time:{end - since}.")
        
        save_path = os.path.join(save_dir, f"fake_{i:02d}.h5")
        fp = h5py.File(save_path, 'w')
        fp.attrs['n_scales'] = config.n_scales
        for j, fake_ in enumerate(fake_list):
            fake_ = fake_.detach().cpu().numpy()[0, 0]
            if config.bin:
                fake_ = fake_ > 0.5
            fp.create_dataset(f'scale{config.n_scales - 1 - j}', data=fake_, compression=9)
        fp.close()


if __name__ == '__main__':
    main()
