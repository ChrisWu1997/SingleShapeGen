from config import get_config
from model import SSGmodel
from utils.data_utils import load_data_fromH5


def main():
    # create experiment config containing all hyperparameters
    cfg = get_config('train')

    # create model
    mymodel = SSGmodel(cfg)

    # load from checkpoint if provided
    if cfg.ckpt is not None:
        mymodel.load_ckpt(cfg.ckpt)
        mymodel.scale += 1

    # load training data
    real_data_list = load_data_fromH5(cfg.src_path)

    # training
    mymodel.train(real_data_list)


if __name__ == '__main__':
    main()
