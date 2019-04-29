from delira import get_backends
import numpy as np


if "TORCH" in get_backends():
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    from deliravision.utils import make_onehot_torch

    class BFocalLossPyTorch(torch.nn.Module):
        """
        Focal loss for binary case without(!) logit

        """

        def __init__(self, alpha=None, gamma=2, reduction='elementwise_mean'):
            """
            Implements Focal Loss for binary class case

            Parameters
            ----------
            alpha : float
                alpha has to be in range [0,1], assigns class weight
            gamma : float
                focusing parameter
            reduction : str
                Specifies the reduction to apply to the output: ‘none’ |
                ‘elementwise_mean’ | ‘sum’. ‘none’: no reduction will be applied,
                ‘elementwise_mean’: the sum of the output will be divided by the
                number of elements in the output, ‘sum’: the output will be summed
            (further information about parameters above can be found in pytorch
            documentation)

            Returns
            -------
            torch.Tensor
                loss value

            """
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, p, t):
            """
            Compute Focal Loss with logits

            Parameters
            ----------
            p : torch.Tensor
                prediction
            t : torch.Tensor
                target

            Returns
            -------
            torch.Tensor
                result
            """
            bce_loss = F.binary_cross_entropy(p, t, reduction='none')

            if self.alpha is not None:
                # create weights for alpha
                alpha_weight = torch.ones(t.shape, device=p.device) * \
                    self.alpha
                alpha_weight = torch.where(torch.eq(t, 1.),
                                           alpha_weight, 1 - alpha_weight)
            else:
                alpha_weight = torch.Tensor([1]).to(p.device)

            # create weights for focal loss
            focal_weight = 1 - torch.where(torch.eq(t, 1.), p, 1 - p)
            focal_weight.pow_(self.gamma)
            focal_weight.to(p.device)

            # compute loss
            focal_loss = focal_weight * alpha_weight * bce_loss

            if self.reduction == 'elementwise_mean':
                return torch.mean(focal_loss)
            if self.reduction == 'none':
                return focal_loss
            if self.reduction == 'sum':
                return torch.sum(focal_loss)
            raise AttributeError('Reduction parameter unknown.')

    class BFocalLossWithLogitsPyTorch(torch.nn.Module):
        """
        Focal loss for binary case WITH logit

        """

        def __init__(self, alpha=None, gamma=2, reduction='elementwise_mean'):
            """
            Implements Focal Loss for binary class case

            Parameters
            ----------
            alpha : float
                alpha has to be in range [0,1], assigns class weight
            gamma : float
                focusing parameter
            reduction : str
                Specifies the reduction to apply to the output: ‘none’ |
                ‘elementwise_mean’ | ‘sum’. ‘none’: no reduction will be applied,
                ‘elementwise_mean’: the sum of the output will be divided by the
                number of elements in the output, ‘sum’: the output will be summed
            (further information about parameters above can be found in pytorch
            documentation)

            Returns
            -------
            torch.Tensor
                loss value

            """
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, p, t):
            """
            Compute Focal Loss with logits

            Parameters
            ----------
            p : torch.Tensor
                prediction
            t : torch.Tensor
                target

            Returns
            -------
            torch.Tensor
                result
            """
            bce_loss = F.binary_cross_entropy_with_logits(
                p, t, reduction='none')

            p = torch.sigmoid(p)

            if self.alpha is not None:
                # create weights for alpha
                alpha_weight = torch.ones_like(t) * self.alpha
                alpha_weight = torch.where(torch.eq(t, 1.),
                                           alpha_weight, 1 - alpha_weight)
            else:
                alpha_weight = torch.Tensor([1]).to(p.device)

            # create weights for focal loss
            focal_weight = 1 - torch.where(torch.eq(t, 1.), p, 1 - p)
            focal_weight.pow_(self.gamma)
            focal_weight.to(p.device)

            # compute loss
            focal_loss = focal_weight * alpha_weight * bce_loss

            if self.reduction == 'elementwise_mean':
                return torch.mean(focal_loss)
            if self.reduction == 'none':
                return focal_loss
            if self.reduction == 'sum':
                return torch.sum(focal_loss)
            raise AttributeError('Reduction parameter unknown.')


    class SoftDiceLossPyTorch(nn.Module):
        def __init__(self, square_nom=False, square_denom=False, weight=None):
            super().__init__()
            self.square_nom = square_nom
            self.square_denom = square_denom
            if weight is not None:
                self.weight = np.array(weight)
            else:
                self.weight = None

        def forward(self, inp, target):
            n_classes = inp.shape[1]
            target_onehot = make_onehot_torch(target, n_classes=n_classes)

            if self.square_nom:
                nom = torch.sum((inp * target_onehot.float())**2, dim=
                                tuple(range(2, inp.dim())))
            else:
                nom = torch.sum(inp * target_onehot.float(), dim=
                                tuple(range(2, inp.dim())))

            if self.square_denom:
                i_sum = torch.sum(inp**2, dim=
                                  tuple(range(2, inp.dim())))
                t_sum = torch.sum(target_onehot**2, dim=
                                  tuple(range(2, target_onehot.dim())))
            else:
                i_sum = torch.sum(inp, dim=
                                  tuple(range(2, inp.dim())))
                t_sum = torch.sum(target_onehot, dim=
                                  tuple(range(2, target_onehot.dim())))
            denom = i_sum + t_sum.float()
            frac = torch.sum(nom / denom, dim=1)
            if self.weight is not None:
                weight = torch.from_numpy(self.weight).to(dtype=inp.dtype,
                                                          device=inp.device)
                frac = weight * frac
            return ((-2/n_classes) * frac).mean()

# if __name__ == "__main__":
#     loss_fn = SoftDiceLossPyTorch(weight=(0.1, 1, 1))
#     inp = torch.rand(1, 3, 128, 128)
#     target = torch.randint(3, (1, 128, 128))
#     loss = loss_fn(inp, target)
#     print(loss)
