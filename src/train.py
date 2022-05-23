from dataset import generate_3Ddata_multiscale
from common import get_config
from agent import get_agent


def main():
    # create experiment config containing all hyperparameters
    config = get_config('train')

    # create network and training agent
    tr_agent = get_agent(config)

    # # load from checkpoint if provided
    if config.ckpt is not None:
        tr_agent.load_ckpt(config.ckpt)

    # create dataloader
    real_data_list = generate_3Ddata_multiscale(config)

    # training
    tr_agent.train(real_data_list)


if __name__ == '__main__':
    main()
