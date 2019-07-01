from deliravision.models.gans.boundary_seeking.models import Generator, \
    Discriminator
from delira.models import AbstractPyTorchNetwork
import torch


class BoundarySeekingGAN(AbstractPyTorchNetwork):
    def __init__(self, latent_dim, img_shape, generator_cls=Generator,
                 discriminator_cls=Discriminator):
        super().__init__()
        self.generator = generator_cls(latent_dim, img_shape)
        self.discriminator = discriminator_cls(img_shape)
        self._latent_dim = latent_dim

    def forward(self, x: torch.Tensor, z: torch.Tensor = None):
        if z is None:
            z = torch.rand(x.shape(0), self._latent_dim, device=x.device,
                           dtype=x.dtype)

        gen_imgs = self.generator(z)

        discr_real = self.discriminator(x)
        discr_fake = self.discriminator(gen_imgs)

        return {"gen_imgs": gen_imgs, "discr_real": discr_real,
                "discr_fake": discr_fake}

    @staticmethod
    def closure(model, data_dict: dict, optimizers: dict, losses=None,
                metrics=None, fold=0, **kwargs):
        """
        Function which handles prediction from batch, logging, loss calculation
        and optimizer step

        Parameters
        ----------
        model : :class:`AbstractPyTorchNetwork`
            model to forward data through
        data_dict : dict
            dictionary containing the data
        optimizers : dict
            dictionary containing all optimizers to perform parameter update
        losses : dict
            Functions or classes to calculate losses
        metrics : dict
            Functions or classes to calculate other metrics; won't be used here
        fold : int
            Current Fold in Crossvalidation (default: 0)
        kwargs : dict
            additional keyword arguments

        Returns
        -------
        dict
            Metric values (with same keys as input dict metrics)
        dict
            Loss values (with same keys as input dict losses)
        dict
            Arbitrary number of predictions

        """

        metric_vals, loss_vals = {}, {}

        preds = model(data_dict["data"])

        loss_gen = losses["boundary_seeking"](preds["discr_fake"], True)
        loss_vals["boundary_seeking_generator"] = loss_gen.item()

        optimizers["generator"].zero_grad()
        loss_gen.backward(retain_graph=True)
        optimizers["generator"].step()

        real_loss = losses["discriminator"](preds["discr_real"], True)
        fake_loss = losses["discriminator"](preds["discr_fake"], True)
        loss_vals["discr_real"] = real_loss.item()
        loss_vals["discr_fake"] = fake_loss.item()

        discr_loss = 0.5 * (real_loss + fake_loss)
        loss_vals["discr_total"] = discr_loss.item()

        optimizers["discriminator"].zero_grad()
        discr_loss.backward()
        optimizers["discriminator"].step()

        # zero gradients again just to make sure, gradients aren't carried to
        # next iteration (won't affect training since gradients are zeroed
        # before every backprop step, but would result in way higher memory
        # consumption)
        for k, v in optimizers.items():
            v.zero_grad()

        # return values
        return metric_vals, loss_vals, {k: v.detach()
                                        for k, v in preds.items()}

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        return {"data": batch["data"].to(torch.float).to(input_device)}