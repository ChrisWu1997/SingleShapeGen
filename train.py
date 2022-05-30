from config import Config
from model import SSGmodel
from utils.data_utils import load_data_fromH5


def main():
    # create experiment config containing all hyperparameters
    cfg = Config('train')

    # create model
    ssg_model = SSGmodel(cfg)

    # load from checkpoint if provided
    if cfg.ckpt is not None:
        ssg_model.load_ckpt(cfg.ckpt)
        ssg_model.scale += 1

    # load training data
    real_data_list = load_data_fromH5(cfg.src_path)

    # training
    ssg_model.train(real_data_list)


if __name__ == '__main__':
    main()
