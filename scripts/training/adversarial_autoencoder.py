from deliravision.models.gans import AdversarialAutoEncoderPyTorch
from deliravision.models.gans.utils import create_optims_gan
from deliravision.losses.gans import AdversarialLoss
import torch
from delira.data_loading import TorchvisionClassificationDataset, \
    BaseDataManager
from delira.training import PyTorchExperiment, Parameters


def train_aae(params, dset, save_path, img_size):

    dset_train = TorchvisionClassificationDataset(dset,
                                                  img_shape=(img_size, img_size))
    dset_val = TorchvisionClassificationDataset(dset, train=False)

    params.fixed.model.img_shape = dset_train[0]["data"].shape[1:]

    mgr_train = BaseDataManager(dset_train, params.nested_get("batchsize"),
                                transforms=None, n_process_augmentation=4)

    mgr_val = BaseDataManager(dset_val, params.nested_get("batchsize"),
                              transforms=None, n_process_augmentation=4)

    exp = PyTorchExperiment(params, model_cls=AdversarialAutoEncoderPyTorch,
                            n_epochs=params.nested_get("n_epochs"),
                            save_path=save_path,
                            optim_builder=create_optims_gan)

    return exp.run(mgr_train, mgr_val)


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--batchsize", type=int, default=64,
                        help="Batchsize")
    parser.add_argument("-e", "--epochs", type=int, default=100,
                        help="Number of Epochs to train")
    parser.add_argument("-s", "--savepath", type=str, default=os.getcwd(),
                        help="Path the experiment checkpoints should be saved "
                             "to")
    parser.add_argument("-d", "--dataset", type=str, default="mnist",
                        help="The torchvision dataset to use")
    parser.add_argument("-i", "--img_size", type=int, default=28,
                        help="Image Size")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")

    args = parser.parse_args()

    params = Parameters(fixed_params={
        "model": {"latent_dim": 100},
        "training": {
            "optim_cls": torch.optim.Adam,
            "optim_params": {"lr": args.lr, "betas": (args.b1, args.b2)},
            "batchsize": args.batchsize,
            "n_epochs": args.epochs,
            "losses": {"adversarial": AdversarialLoss(),
                       "pixelwise": torch.nn.L1Loss()}
        }
    })

    train_aae(params, args.dataset, args.savepath, args.img_size)

