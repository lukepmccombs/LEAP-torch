from torch import nn


def check_module(genome):
    """
    Checks that the provided genome is a pytorch module.

    :param genome: possibly a pytorch module
    """
    assert isinstance(genome, nn.Module),\
        "Genome must be of type torch.nn.Module"