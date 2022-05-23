import os
import h5py
import time
from utils import ensure_dir
from dataset import generate_3Ddata_multiscale
from common import get_config
from model import get_agent
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates


def make_coord(H, W, D, device, normalize=True):
    """ Make coordinates at grid centers.
    """
    xs = torch.arange(H, device=device).float() 
    ys = torch.arange(W, device=device).float()
    zs = torch.arange(D, device=device).float()
    if normalize:
        xs = xs / (H - 1) * 2 - 1 # (-1, 1)
        ys = ys / (W - 1) * 2 - 1 # (-1, 1)
        zs = zs / (D - 1) * 2 - 1 # (-1, 1)

    coords = torch.stack(torch.meshgrid(xs, ys, zs), dim=-1)
    return coords


def main():
    # create experiment config containing all hyperparameters
    config = get_config('test')

    # create network and training agent
    tr_agent = get_agent(config)

    # load from checkpoint if provided
    tr_agent.load_ckpt(config.n_scales)

    real_data_list = generate_3Ddata_multiscale(config)
    tr_agent.set_real_data(real_data_list)

    x_list = [1, 8] # [:config.n_samples]
    out_name = f"outHSx{x_list[-1]}_{config.test_mode}"
    if config.bin:
        out_name += "_bin"
    if config.seq:
        out_name += "_seq"
    out_name += f"_r{config.resize[0]}x{config.resize[1]}x{config.resize[2]}"    
    save_dir = os.path.join(config.exp_dir, out_name)
    ensure_dir(save_dir)

    out_1x = None

    init_inp = tr_agent.template
    torch.manual_seed(123)
    init_noise = tr_agent.draw_init_noise(config.test_mode)
    real_shapes = tr_agent.real_shapes[:config.n_scales - 1 + 1]
    noises_list = tr_agent.draw_noises_list(config.test_mode, config.n_scales - 1)

    for i in range(len(x_list)):
        since = time.time()

        query_shape = real_shapes[-1]
        query_shape = [x * x_list[i] for x in query_shape]
        coords = make_coord(*query_shape, tr_agent.device)
        print(coords.shape)

        with torch.no_grad():
            out = tr_agent.netG(init_noise, init_inp, real_shapes, noises_list, config.test_mode, coords=coords)
        # if i == 0:
        #     out_1x = (out > 0.5).float()
        fake_list = [out]

        end = time.time()
        print(f"{i}. time:{end - since}.")
        
        save_path = os.path.join(save_dir, f"fake_{i:02d}_x{x_list[i]}.h5")
        fp = h5py.File(save_path, 'w')
        fp.attrs['n_scales'] = config.n_scales
        for j, fake_ in enumerate(fake_list):
            fake_ = fake_.detach().cpu().numpy()[0, 0]
            if config.bin:
                fake_ = fake_ > 0.5
            fp.create_dataset(f'scale{config.n_scales - 1 - j}', data=fake_, compression=9)
        fp.close()

        # # interpolated
        # out_iterp = F.interpolate(out_1x, query_shape, mode='trilinear')
        # out_iterp = out_iterp.detach().cpu().numpy()[0, 0] > 0.5
        # save_path = os.path.join(save_dir, f"fake_{i:02d}iterp.h5")
        # fp = h5py.File(save_path, 'w')
        # fp.attrs['n_scales'] = config.n_scales
        # fp.create_dataset(f'scale{config.n_scales - 1}', data=out_iterp, compression=9)
        # fp.close()

        # # spline interpolation
        # int_coords = make_coord(*query_shape, "cpu", normalize=False).cpu().numpy().reshape(-1, 3).transpose()
        # out_1x_arr = out_1x.detach().cpu().numpy()[0, 0]
        # print(out_1x_arr.shape, int_coords.shape)
        # out_spline = map_coordinates(out_1x_arr, int_coords) > 0.5
        # out_spline = out_spline.reshape(*query_shape)
        # save_path = os.path.join(save_dir, f"fake_{i:02d}spline.h5")
        # fp = h5py.File(save_path, 'w')
        # fp.attrs['n_scales'] = config.n_scales
        # fp.create_dataset(f'scale{config.n_scales - 1}', data=out_spline, compression=9)
        # fp.close()


if __name__ == '__main__':
    main()
    # cs = make_coord(2, 3, 2, "cpu")
    # print(cs.shape)
    # print(cs[0, :, :, [1, 2]])
    # print(cs[1, :, :, [1, 2]])
