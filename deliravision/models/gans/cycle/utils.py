import itertools


def create_optims_cycle_gan(model, optim_cls, **optim_params):
    """
    Creates all necessary optimizers to train the cycle GAN (one per
    discriminator and a combined optimizer for both generators)

    Parameters
    ----------
    model : :class:`delira.models.AbstractPyTorchModel`
        the model to optimize
    optim_cls : subclass of :class:`torch.optim.Optimizer`
        the optimizer class to use
    **optim_params :
        additional keyword arguments accepted by the :param:`optim_cls`

    Returns
    -------
    dict
        a dictionary containing all the optimizers

    """
    return {
        "generator": optim_cls(itertools.chain(
            model.generator_a.parameters(),
            model.generator_b.parameters()), **optim_params),
        "discriminator_a": optim_cls(model.discriminator_a.parameters(),
                                     **optim_params),
        "discriminator_b": optim_cls(model.discriminator_b.parameters(),
                                     **optim_params)
    }