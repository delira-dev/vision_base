from delira import get_backends as __get_backends

__all__ = []

if "TORCH" in __get_backends():

    from .segmentation import BaseSegmentationPyTorchNetwork
    from .classification import BaseClassificationPyTorchNetwork

    __all__ += [
        "BaseSegmentationPyTorchNetwork",
        "BaseClassificationPyTorchNetwork"
    ]