from delira import  get_backends as __get_backends


__all__ = []

if "TORCH" in __get_backends():

    from .torchvision_classification import TorchVisionClassificationDataset, \
        TorchVisionImageFolder, TorchVisionCIFAR10, TorchVisionCIFAR100, \
        TorchVisionDatasetFolder, TorchVisionEMNIST, TorchVisionFakeData, \
        TorchVisionFashionMNIST, TorchVisionImageNet, TorchVisionKMNIST, \
        TorchVisionLSUN, TorchVisionMNIST, TorchVisionSTL10, TorchVisionSVHN

    __all__ += [
        'TorchVisionCIFAR10',
        'TorchVisionCIFAR100',
        'TorchVisionClassificationDataset',
        'TorchVisionDatasetFolder',
        'TorchVisionEMNIST',
        'TorchVisionFakeData',
        'TorchVisionFashionMNIST',
        'TorchVisionImageFolder',
        'TorchVisionImageNet',
        'TorchVisionKMNIST',
        'TorchVisionLSUN',
        'TorchVisionMNIST',
        'TorchVisionSTL10',
        'TorchVisionSVHN'
    ]
