from delira.models import AbstractPyTorchNetwork
from deliravision.models.gans.boundary_equilibrium.models import \
    Generator, Discriminator

import torch


class BoundaryEquilibriumGAN(AbstractPyTorchNetwork):
    """
    A basic implementation of Boundary Equilibrium Generative Adversarial
    Networks with variable generator and discriminator networks

    See Also
    --------
    `Paper <https://arxiv.org/abs/1703.10717>`_

    """
    def __init__(self, n_channels, latent_dim, img_size,
                 generator_cls=Generator, discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        n_channels : int
            the number of image channels
        latent_dim : int
            the size of the latent dimension
        img_size : int
            the size of the squared input images (per side)
        generator_cls :
            subclass of :class:`torch.nn.Module` implementing the actual
            generator topology
        discriminator_cls :
            subclass of :class:`torch.nn.Module` implementing the actual
            discriminator topology

        """
        super().__init__()
        self.generator = generator_cls(n_channels=n_channels,
                                       latent_dim=latent_dim,
                                       img_size=img_size)

        self.discriminator = discriminator_cls(n_channels=n_channels,
                                               img_size=img_size)

        self._latent_dim = latent_dim

    def forward(self, x: torch.Tensor, noise=None):
        """
        Forwards a real image batch and an image batch generated from (sampled)
        noise through the discriminator

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the batch of real images
        noise : :class:`torch.Tensor`
            the noise vector to generate images from;
            If None: noise vector will be sampled from normal distrbution

        Returns
        -------
        dict
            a dictionary containing all relevant predictions

        """
        if noise is None:
            noise = torch.rand(x.size(0),
                               self._latent_dim).to(x.dtype).to(x.device)

        gen_imgs = self.generator(noise)

        return {"gen_imgs": gen_imgs, "discr_real": self.discriminator(x),
                "discr_fake": self.discriminator(gen_imgs)}

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
            Functions or classes to calculate other metrics
        fold : int
            Current Fold in Crossvalidation (default: 0)
        kwargs : dict
            additional keyword arguments

        Returns
        -------
        dict
            Metric values (with same keys as input dict metrics);
            Will always be empty here
        dict
            Loss values (with same keys as input dict losses)
        dict
            Arbitrary number of predictions

        """

        loss_vals, metric_vals = {}, {}

        predictions = model(data_dict["labels"])

        loss_gen = (predictions["discr_fake"]-data_dict["label"]).abs().mean()
        loss_vals["gen_total"] = loss_gen.item()

        optimizers["generator"].zero_grad()
        loss_gen.backward(retain_graph=True)
        optimizers["generator"].step()

        discr_loss, discr_loss_real, discr_loss_fake = losses["began"](
            predictions["discr_real"],
            data_dict["label"],
            predictions["discr_fake"],
            predictions["gen_imgs"])

        loss_vals["discr_real"] = discr_loss_real.item()
        loss_vals["discr_fake"] = discr_loss_fake.item()
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

        return metric_vals, loss_vals, {k: v.detach()
                                        for k, v in predictions.items()}

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        return {"data": batch["data"].to(torch.float).to(input_device)}