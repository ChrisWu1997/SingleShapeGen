from .model import SSGmodelTP


def get_model(config):
    if config.G_struct == "triplane":
        return SSGmodelTP(config)
    elif config.G_struct == "conv3d":
        pass
    else:
        raise NotImplementedError
