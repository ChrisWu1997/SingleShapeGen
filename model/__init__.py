from .model_base import SSGmodelBase
from .models import SSGmodelTriplane, SSGmodelConv3D


def get_model(config):
    if config.G_struct == "triplane":
        return SSGmodelTriplane(config)
    elif config.G_struct == "conv3d":
        return SSGmodelConv3D(config)
    else:
        raise NotImplementedError
