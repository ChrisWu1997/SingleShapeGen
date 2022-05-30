import os
import argparse
import json
import shutil
from utils.file_utils import ensure_dirs


def get_config(phase):
    config = Config(phase)
    return config


class Config(object):
    """Base class of Config, provide necessary hyperparameters. 
    """
    def __init__(self, phase):
        self.is_train = phase == "train"

        # init hyperparameters and parse from command-line
        parser, args = self.parse()

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        # experiment paths
        self.exp_dir = os.path.join(self.proj_dir, self.tag)
        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')

        # load saved config if not training
        if not self.is_train:
            assert os.path.exists(self.exp_dir)
            config_path = os.path.join(self.exp_dir, 'config.json')
            print(f"Load saved config from {config_path}")
            with open(config_path, 'r') as f:
                saved_args = json.load(f)
            for k, v in saved_args.items():
                if not hasattr(self, k):
                    self.__setattr__(k, v)
            return

        # re-mkdir if re-training
        if phase == "train" and args.ckpt is None and os.path.exists(self.exp_dir):
            response = input('Experiment log/model already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)
        ensure_dirs([self.log_dir, self.model_dir])

        # save this configuration
        if self.is_train:
            with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)
            copy_code_dir = os.path.join(self.exp_dir, "code") # FIXME: remove when release
            ensure_dirs(copy_code_dir)
            os.system("cp *.py {}".format(copy_code_dir))

    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()
        
        # basic configuration
        self._add_basic_config_(parser)

        if self.is_train:
            # dataset configuration
            self._add_dataset_config_(parser)
            # model configuration
            self._add_network_config_(parser)
            # training or testing configuration
            self._add_training_config_(parser)
        else:
            self._add_testing_config_(parser)

        args = parser.parse_args()
        return parser, args

    def _add_basic_config_(self, parser):
        """add general hyperparameters"""
        group = parser.add_argument_group('basic')
        group.add_argument('--proj_dir', type=str, default="project_log", help="a folder where models and logs will be saved")
        group.add_argument('--tag', type=str, required=True, help="name of this experiment")
        group.add_argument('-g', '--gpu_ids', type=str, default=0, help="gpu to use, e.g. 0  0,1,2. CPU not supported.")

    def _add_dataset_config_(self, parser):
        """add hyperparameters for dataset configuration"""
        group = parser.add_argument_group('data')
        group.add_argument('-s', '--src_path', type=str, help='source data path', default=None)

    def _add_network_config_(self, parser):
        """add hyperparameters for network architecture"""
        group = parser.add_argument_group('network')
        group.add_argument("--D_nc", type=int, default=32, help="number of conv channels for discriminator")
        group.add_argument("--D_layers", type=int, default=3, help="number of conv layers for discriminator")
        group.add_argument("--G_nc", type=int, default=32, help="number of conv channels for generator")
        group.add_argument("--G_layers", type=int, default=4, help="number of conv layers for generator")
        group.add_argument("--mlp_dim", type=int, default=32, help="number of hidden features for MLP")
        group.add_argument("--mlp_layers", type=int, default=0, help="number of hidden layers for MLP")
        group.add_argument("--pool_dim", type=int, default=8, help="average pooling dimension")
        group.add_argument("--feat_dim", type=int, default=32, help="tri-plane feature dimension")

        # group.add_argument('--use_norm', type=int, default=1, help="use BN")
        # group.add_argument('--use_norm', action='store_true', help='enable normalization layer')
        group.add_argument('--no_norm', dest='use_norm', action='store_false', help='disable normalization layer')
        group.set_defaults(use_norm=True) # FIXME: simpler option for python > 3.9

    def _add_training_config_(self, parser):
        """training configuration"""
        group = parser.add_argument_group('training')
        group.add_argument('--ckpt', type=int, default=None, help="restore checkpoint at scale x")
        group.add_argument('--save_frequency', type=int, default=3000, help="save models every x iterations")
        group.add_argument('--vis_frequency', type=int, default=200, help="visualize output every x iterations")
        group.add_argument('--n_iters', type=int, default=2000, help='number of iterations to train per scale')
        group.add_argument('--lr_g', type=float, default=0.0001, help='G learning rate, default=0.0005')
        group.add_argument('--lr_d', type=float, default=0.0001, help='D learning rate, default=0.0005')
        group.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
        group.add_argument('--Gsteps',type=int, default=3, help='Generator inner steps per iteration')
        group.add_argument('--Dsteps',type=int, default=3, help='Discriminator inner steps per iteration')
        group.add_argument('--lambda_grad',type=float, default=0.1, help='gradient penelty weight')
        group.add_argument('--alpha',type=float, default=10, help='reconstruction loss weight')
        group.add_argument('--base_noise_amp', type=float, default=0.1, help='base noise amplifier')
        group.add_argument('--train_depth',type=int, default=1, help='number of concurrent training depth')
        group.add_argument('--lr_sigma',type=float,default=0.1, help='learning rate scaling for lower scale when train_depth > 1')

    def _add_testing_config_(self, parser):
        """testing configuration"""
        group = parser.add_argument_group('testing')
        group.add_argument('--ckpt', type=int, default=None, help="use checkpoint at scale x. By default, use the highest scale.")
        group.add_argument('--mode', type=str, default='rand', choices=['rand', 'rec', 'interp'], help="inference mode")
        group.add_argument("--resize", nargs="*", type=float, default=[1, 1, 1], help="resize factor along each axis")
        group.add_argument('--n_samples', type=int, default=1, help="number of samples to generate")
        # group.add_argument('--bin', action='store_true', help="binarize the output so to save as boolean type")
        group.add_argument('--no_bin', dest='bin', action='store_false', help='save non-binary output')
        group.set_defaults(bin=True)
        # group.add_argument('--seq', action='store_true', help="save result of each scale")
