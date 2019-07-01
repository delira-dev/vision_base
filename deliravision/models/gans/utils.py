from delira.models import AbstractPyTorchNetwork


def create_optims_gan(model: AbstractPyTorchNetwork, optim_cls, **optim_params):
    return {"generator": optim_cls(model.generator.parameters(),
                                   **optim_params),
            "discriminator": optim_cls(model.discriminator.parameters(),
                                       **optim_params)
            }
