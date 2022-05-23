import os
import h5py
import time
from utils import ensure_dir
from dataset import generate_3Ddata_multiscale
from common import get_config
from model import get_agent
import torch


def prepare():
    # create experiment config containing all hyperparameters
    config = get_config('test')

    # create network and training agent
    tr_agent = get_agent(config)

    # load from checkpoint if provided
    tr_agent.load_ckpt(config.n_scales)

    real_data_list = generate_3Ddata_multiscale(config)
    tr_agent.set_real_data(real_data_list)

    out_name = f"interpSrc_{config.test_mode}_n{config.n_samples}"
    if config.bin:
        out_name += "_bin"
    out_name += f"_r{config.resize[0]}x{config.resize[1]}x{config.resize[2]}"    
    save_dir = os.path.join(config.exp_dir, out_name)
    ensure_dir(save_dir)

    for i in range(config.n_samples):
        since = time.time()
        
        init_inp = tr_agent.template
        init_noise = tr_agent.draw_init_noise(config.test_mode)
        real_shapes = tr_agent.real_shapes[:config.n_scales - 1 + 1]
        noises_list = tr_agent.draw_noises_list(config.test_mode, config.n_scales - 1)
        with torch.no_grad():
            fake_ = tr_agent.netG(init_noise, init_inp, real_shapes, noises_list, config.test_mode)
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
            init_noise = init_noise.detach().cpu().numpy()
            fp.create_dataset(f'init_noise', data=init_noise, compression=9)
        fp.close()


def main():
    # create experiment config containing all hyperparameters
    config = get_config('test')

    # create network and training agent
    tr_agent = get_agent(config)

    # load from checkpoint if provided
    tr_agent.load_ckpt(config.n_scales)

    real_data_list = generate_3Ddata_multiscale(config)
    tr_agent.set_real_data(real_data_list)

    out_name = f"extra_{config.test_mode}_n{config.n_samples}"
    if config.bin:
        out_name += "_bin"
    if config.seq:
        out_name += "_seq"
    out_name += f"_r{config.resize[0]}x{config.resize[1]}x{config.resize[2]}"    
    out_dir = os.path.join(config.exp_dir, out_name)
    ensure_dir(out_dir)


    init_inp = tr_agent.template
    # torch.manual_seed(123)
    torch.manual_seed(1234)
    is_random = True
    if not is_random:
        # load
        # src_dir = "/local/cg/rundi/workspace/ssg_code/project_log/vtpfmsV1_acropolisFm15_res256s8dep1/interp_rand_n10_bin_r1x1x1/pair9"
        # src_dir = "/local/cg/rundi/workspace/ssg_code/project_log/vtpfmsV1_superterrainFm15_res256s9dep1/interp_rand_n5_bin_r1x1x1/pair1"
        # src_dir = "/local/cg/rundi/workspace/ssg_code/project_log/vtpfmsV1_table1Fm15_res256s8dep1/interp_rand_n5_bin_r1x1x1/pair2"
        # src_dir = "/local/cg/rundi/workspace/ssg_code/project_log/vtpfmsV1_boulderstone_res256s8dep1/interp_rand_n10_bin_r1x1x1/pair7"
        src_dir = "/local/cg/rundi/workspace/ssg_code/project_log/vtpfmsV1_wood03Fm15_res256s8dep1/interp_rand_n10_bin_r1x1x1/pair1"
        with h5py.File(os.path.join(src_dir, 'fake_00.h5'), 'r') as fp:
            init_noise1_load = torch.tensor(fp['init_noise'][:], dtype=torch.float).cuda()
        with h5py.File(os.path.join(src_dir, 'fake_04.h5'), 'r') as fp:
            init_noise2_load = torch.tensor(fp['init_noise'][:], dtype=torch.float).cuda()

    alpha_list = [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    print(alpha_list)

    n_try = config.n_samples if is_random else 1
    for k in range(n_try):
        # random
        if is_random:
            init_noise1 = tr_agent.draw_init_noise(config.test_mode)
            init_noise2 = tr_agent.draw_init_noise(config.test_mode)
        else:
            init_noise1 = init_noise1_load
            init_noise2 = init_noise2_load

        real_shapes = tr_agent.real_shapes[:config.n_scales - 1 + 1]
        noises_list = tr_agent.draw_noises_list(config.test_mode, config.n_scales - 1)
        for i in range(len(alpha_list)):
            since = time.time()

            alpha = alpha_list[i]
            print("alpha=", alpha)
            init_noise = (1 - alpha) * init_noise1 + alpha * init_noise2
            with torch.no_grad():
                out = tr_agent.netG(init_noise, init_inp, real_shapes, noises_list, config.test_mode)
            fake_list = [out]

            end = time.time()
            print(f"{i}. time:{end - since}.")
            
            save_dir = os.path.join(out_dir, f'pair{k}')
            ensure_dir(save_dir)
            save_path = os.path.join(save_dir, f"fake_{i:02d}_alpha{alpha}.h5")
            fp = h5py.File(save_path, 'w')
            fp.attrs['n_scales'] = config.n_scales
            for j, fake_ in enumerate(fake_list):
                fake_ = fake_.detach().cpu().numpy()[0, 0]
                if config.bin:
                    fake_ = fake_ > 0.5
                fp.create_dataset(f'scale{config.n_scales - 1 - j}', data=fake_, compression=9)
                init_noise = init_noise.detach().cpu().numpy()
                fp.create_dataset(f'init_noise', data=init_noise, compression=9)
            fp.close()


if __name__ == '__main__':
    # prepare()
    main()
