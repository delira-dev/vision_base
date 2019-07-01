def create_optims_bicycle_gan(model, optim_cls, **optim_params):
    """
    Utility function to create generators suitable for a bicycle GAN
    Parameters
    ----------
    model : :class:`deliravision.models.gans.BicycleGAN`
        the model to optimize
    optim_cls :
        the optimizer class to use; must be subclass of
        :class:`torch.optim.Optimizer`
    **optim_params :
        additional optimizer parameters

    Returns
    -------
    dict
        the created optimizers

    """
    return {
        "encoder": optim_cls(model.encoder.parameters(), **optim_params),
        "generator": optim_cls(model.generator.parameters(), **optim_params),
        "discriminator_lr": optim_cls(model.discr_lr.parameters(),
                                      **optim_params),
        "discriminator_vae": optim_cls(model.discr_vae.parameters(),
                                       **optim_params)
    }
