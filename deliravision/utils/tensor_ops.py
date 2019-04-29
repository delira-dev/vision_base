from delira import get_backends
import numpy as np

def make_onehot_npy(labels, n_classes):
    labels = labels.reshape(-1).astype(np.uint8)
    return np.eye(n_classes)[labels]

if "TORCH" in get_backends():

    import torch

    def make_onehot_torch(labels, n_classes):
        labels = labels.view(-1, 1).long()

        labels_onehot = torch.zeros(labels.size(0), n_classes,
                                    device=labels.device, dtype=torch.float)
        labels_onehot.scatter_(1, labels, 1)
        return labels_onehot
