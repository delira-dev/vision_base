from delira import get_backends
import numpy as np


def make_onehot_npy(labels, n_classes):
    labels = labels.reshape(-1).astype(np.uint8)
    return np.eye(n_classes)[labels]


if "TORCH" in get_backends():
    import torch

    def make_onehot_torch(labels, n_classes):
        idx = labels.to(dtype=torch.long)

        new_shape = list(labels.unsqueeze(dim=1).shape)
        new_shape[1] = n_classes
        labels_onehot = torch.zeros(*new_shape, device=labels.device,
                                    dtype=labels.dtype)
        labels_onehot.scatter_(1, idx.unsqueeze(dim=1), 1)
        return labels_onehot
