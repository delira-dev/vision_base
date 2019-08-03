import numpy as np


def make_onehot_npy(labels, n_classes):
    labels = labels.reshape(-1).astype(np.uint8)
    return np.eye(n_classes)[labels]
