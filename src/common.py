import os
import argparse
import json
import shutil
from utils import ensure_dirs


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

        # experiment paths
        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        if phase == "train" and args.ckpt is None and os.path.exists(self.exp_dir):
            response = input('Experiment log/model already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)

        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        ensure_dirs([self.log_dir, self.model_dir])

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        # save this configuration
        if self.is_train:
            with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)
            copy_code_dir = os.path.join(self.exp_dir, "code")
            ensure_dirs(copy_code_dir)
            os.system("cp *.py {}".format(copy_code_dir))

    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()
        
        # basic configuration
        self._add_basic_config_(parser)

        # dataset configuration
        self._add_dataset_config_(parser)

        # model configuration
        self._add_network_config_(parser)

        # training or testing configuration
        self._add_training_config_(parser)

        # additional parameters if needed
        if not self.is_train:
            self._add_testing_config_(parser)


        args = parser.parse_args()
        return parser, args

    def _add_basic_config_(self, parser):
        """add general hyperparameters"""
        group = parser.add_argument_group('basic')
        group.add_argument('--proj_dir', type=str, default="/local/cg/rundi/project_log/ssg", help="path to project folder where models and logs will be saved")
        group.add_argument('--data_root', type=str, default="your-data-root", help="path to source data folder")
        group.add_argument('--exp_name', type=str, default=os.getcwd().split('/')[-1], help="name of this experiment")
        group.add_argument('-g', '--gpu_ids', type=str, default=0, help="gpu to use, e.g. 0  0,1,2. CPU not supported.")

    def _add_dataset_config_(self, parser):
        """add hyperparameters for dataset configuration"""
        group = parser.add_argument_group('dataset')
        # group.add_argument('--batch_size', type=int, default=1, help="batch size")
        # group.add_argument('--num_workers', type=int, default=0, help="number of workers for data loading")
        group.add_argument('--res', type=int, help='resolution', default=128)
        group.add_argument('--path', type=str, help='source shape path', default=None)
        group.add_argument('--factor', type=float, help='scaling factor', default=0.75)
        group.add_argument('--n_scales', type=int, help='number of scales', default=7)

    def _add_network_config_(self, parser):
        """add hyperparameters for network architecture"""
        group = parser.add_argument_group('network')
        group.add_argument("--D_nc", type=int, default=32, help="number of conv channels for discriminator")
        group.add_argument("--D_layers", type=int, default=3, help="number of conv layers for discriminator")
        group.add_argument("--G_nc", type=int, default=32, help="number of conv channels for generator")
        group.add_argument("--G_layers", type=int, default=4, help="number of conv layers for generator")
        group.add_argument("--mlp_dim", type=int, default=32, help="number of hidden features for MLP")
        group.add_argument("--mlp_layers", type=int, default=0, help="number of hidden layers for MLP")
        group.add_argument("--pool_dim", type=int, default=32, help="average pooling dimension")
        group.add_argument("--feat_dim", type=int, default=32, help="tri-plane feature dimension")

        # group.add_argument('--use_norm', type=int, default=1, help="use BN")
        # group.add_argument('--use_norm', action='store_true', help='enable normalization layer')
        group.add_argument('--no_norm', dest='use_norm', action='store_false', help='disable normalization layer')
        group.set_defaults(use_norm=True) # FIXME: simpler option for python > 3.9

    def _add_training_config_(self, parser):
        """training configuration"""
        group = parser.add_argument_group('training')
        # group.add_argument('--nr_epochs', type=int, default=1000, help="total number of epochs to train")
        # group.add_argument('--continue', dest='cont',  action='store_true', help="continue training from checkpoint")
        group.add_argument('--ckpt', type=int, default=None, required=False, help="desired checkpoint to restore")
        # group.add_argument('--vis', action='store_true', default=False, help="visualize output in training")
        group.add_argument('--save_frequency', type=int, default=3000, help="save models every x epochs")
        # group.add_argument('--val_frequency', type=int, default=10, help="run validation every x iterations")
        group.add_argument('--vis_frequency', type=int, default=200, help="visualize output every x iterations")
        group.add_argument('--n_iters', type=int, default=2000, help='number of epochs to train per scale')
        group.add_argument('--gamma', type=float, help='scheduler gamma', default=0.1)
        group.add_argument('--lr_stepsize', type=int, help='scheduler lr_stepsize', default=10000)
        group.add_argument('--lr_g', type=float, default=0.0001, help='learning rate, default=0.0005')
        group.add_argument('--lr_d', type=float, default=0.0001, help='learning rate, default=0.0005')
        group.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
        group.add_argument('--Gsteps',type=int, help='Generator inner steps',default=3)
        group.add_argument('--Dsteps',type=int, help='Discriminator inner steps',default=3)
        group.add_argument('--lambda_grad',type=float, help='gradient penelty weight',default=0.1)
        group.add_argument('--alpha',type=float, help='reconstruction loss weight',default=10)
        # group.add_argument('--init_noise', type=float, help='initial noise amplifier', default=1.0)
        group.add_argument('--init_noise_amp', type=float, help='initial noise amplifier', default=0.1)

        group.add_argument('--train_depth',type=int,help='number of train depth',default=1)

        group.add_argument('--sigma',type=float, help='',default=0.1)

    def _add_testing_config_(self, parser):
        """testing configuration"""
        group = parser.add_argument_group('testing')
        group.add_argument('--test_mode', type=str, default='rand', choices=['rand', 'rec'])
        group.add_argument("--resize", nargs="*", type=float, default=[1, 1, 1])
        group.add_argument('--n_samples', type=int, default=1)
        group.add_argument('--bin', action='store_true', help="binarize output")
        group.add_argument('--seq', action='store_true', help="save result of each scale")
