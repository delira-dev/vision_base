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
            Implements Focal Loss for binary classification case

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
        Focal loss for binary case WITH logits

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

    class FocalLossPyTorch(nn.Module):
        def __init__(self, alpha=None, gamma=2, reduction="elementwise_mean"):
            super().__init__()
            self.gamma = gamma
            self.nnloss_fn = nn.NLLLoss(weight=alpha, reduction="none")
            self.reduction = reduction

        def forward(self, inp, target):
            n_classes = inp.shape[1]
            inp_log = torch.log(inp)
            nn_loss = self.nnloss_fn(inp_log, target)

            target_onehot = make_onehot_torch(target, n_classes=n_classes)
            focal_weights = ((1 - inp) * target_onehot.to(
                torch.float)).sum(dim=1) ** self.gamma
            focal_loss = focal_weights * nn_loss
            if self.reduction == 'elementwise_mean':
                return torch.mean(focal_loss)
            if self.reduction == 'none':
                return focal_loss
            if self.reduction == 'sum':
                return torch.sum(focal_loss)
            raise AttributeError('Reduction parameter unknown.')

    class FocalLossWithLogitsPyTorch(nn.Module):
        def __init__(self, alpha=None, gamma=2, reduction="elementwise_mean"):
            super().__init__()
            self.gamma = gamma
            self.ce_fn = nn.CrossEntropyLoss(weight=alpha, reduction="none")
            self.reduction = reduction

        def forward(self, inp, target):
            n_classes = inp.shape[1]
            ce_loss = self.ce_fn(inp, target)
            inp = F.softmax(inp, dim=1)

            target_onehot = make_onehot_torch(target, n_classes=n_classes)
            focal_weights = ((1 - inp) * target_onehot.to(
                torch.float)).sum(dim=1) ** self.gamma
            focal_loss = focal_weights * ce_loss
            if self.reduction == 'elementwise_mean':
                return torch.mean(focal_loss)
            if self.reduction == 'none':
                return focal_loss
            if self.reduction == 'sum':
                return torch.sum(focal_loss)
            raise AttributeError('Reduction parameter unknown.')

    class SoftDiceLossPyTorch(nn.Module):
        def __init__(self, square_nom=False, square_denom=False, weight=None,
                     smooth=1., reduction="elementwise_mean", non_lin=None):
            """
            SoftDice Loss

            Parameters
            ----------
            square_nom : bool
                square nominator
            square_denom : bool
                square denominator
            weight : iterable
                additional weighting of individual classes
            smooth : float
                smoothing for nominator and denominator
            """
            super().__init__()
            self.square_nom = square_nom
            self.square_denom = square_denom

            self.smooth = smooth
            if weight is not None:
                self.weight = np.array(weight)
            else:
                self.weight = None

            self.reduction = reduction
            self.non_lin = non_lin

        def forward(self, inp, target):
            """
            Compute SoftDice Loss

            Parameters
            ----------
            inp : torch.Tensor
                prediction
            target : torch.Tensor
                ground truth tensor

            Returns
            -------
            torch.Tensor
                loss
            """
            # number of classes for onehot
            n_classes = inp.shape[1]
            target_onehot = make_onehot_torch(target, n_classes=n_classes)
            # sum over spatial dimensions
            dims = tuple(range(2, inp.dim()))

            # apply nonlinearity
            if self.non_lin is not None:
                inp = self.non_lin(inp)

            # compute nominator
            if self.square_nom:
                nom = torch.sum((inp * target_onehot.float()) ** 2, dim=dims)
            else:
                nom = torch.sum(inp * target_onehot.float(), dim=dims)
            nom = 2 * nom + self.smooth

            # compute denominator
            if self.square_denom:
                i_sum = torch.sum(inp ** 2, dim=dims)
                t_sum = torch.sum(target_onehot ** 2, dim=dims)
            else:
                i_sum = torch.sum(inp, dim=dims)
                t_sum = torch.sum(target_onehot, dim=dims)

            denom = i_sum + t_sum.float() + self.smooth

            # compute loss
            frac = nom / denom

            # apply weight for individual classesproperly
            if self.weight is not None:
                weight = torch.from_numpy(self.weight).to(dtype=inp.dtype,
                                                          device=inp.device)
                frac = weight * frac

            # average over classes
            frac = - torch.mean(frac, dim=1)

            if self.reduction == 'elementwise_mean':
                return torch.mean(frac)
            if self.reduction == 'none':
                return frac
            if self.reduction == 'sum':
                return torch.sum(frac)
            raise AttributeError('Reduction parameter unknown.')

    class TverskyLossPytorch(nn.Module):
        def __init__(self, alpha=0.5, beta=0.5, square_nom=False,
                     square_denom=False, weight=None, smooth=1.,
                     reduction="elementwise_mean", non_lin=None):
            super().__init__()
            self.alpha = alpha
            self.beta = beta

            self.square_nom = square_nom
            self.square_denom = square_denom

            self.smooth = smooth
            if weight is not None:
                self.weight = np.array(weight)
            else:
                self.weight = None

            self.reduction = reduction
            self.non_lin = non_lin

        def forward(self, pred, target):
            n_classes = pred.shape[1]
            dims = tuple(range(2, pred.dim()))

            if self.non_lin is not None:
                pred = self.non_lin(pred)

            target_onehot = make_onehot_torch(target, n_classes=n_classes)
            target_onehot = target_onehot.float()

            tp = pred * target_onehot
            fp = pred * (1 - target_onehot)
            fn = (1 - pred) * target_onehot

            if self.square_nom:
                tp = tp ** 2
            if self.square_denom:
                fp = fp ** 2
                fn = fn ** 2

            # compute nominator
            tp_sum = torch.sum(tp, dim=dims)
            nom = tp_sum + self.smooth

            # compute denominator
            denom = tp_sum + self.alpha * torch.sum(fn, dim=dims) + \
                self.beta * torch.sum(fp, dim=dims) + self.smooth

            # compute loss
            frac = nom / denom

            # apply weights to individual classes
            if self.weight is not None:
                weight = torch.from_numpy(self.weight).to(dtype=pred.dtype,
                                                          device=pred.device)
                frac = weight * frac

            # average over classes
            frac = 1 - torch.mean(frac, dim=1)

            if self.reduction == 'elementwise_mean':
                return torch.mean(-frac)
            if self.reduction == 'none':
                return -frac
            if self.reduction == 'sum':
                return torch.sum(-frac)
            raise AttributeError('Reduction parameter unknown.')

    class FocalTverskyLossPytorch(nn.Module):
        def __init__(self, gamma=1.33, alpha=0.5, beta=0.5, square_nom=False,
                     square_denom=False, weight=None, smooth=1.,
                     reduction="elementwise_mean", non_lin=None):
            super().__init__()
            self.gamma = gamma
            self.tversky_loss_fn = \
                TverskyLossPytorch(alpha, beta, square_nom,
                                   square_denom, weight, smooth,
                                   reduction='none', non_lin=non_lin)
            self.reduction = reduction

        def forward(self, pred, target):
            n_classes = pred.shape[1]
            tversky_loss = self.tversky_loss_fn(pred, target)
            focal_tversky_loss = tversky_loss ** self.gamma

            if self.reduction == 'elementwise_mean':
                return torch.mean(focal_tversky_loss)
            if self.reduction == 'none':
                return focal_tversky_loss
            if self.reduction == 'sum':
                return torch.sum(focal_tversky_loss)
            raise AttributeError('Reduction parameter unknown.')
