from delira import get_backends
from delira.data_loading import AbstractDataset
import numpy as np

if "TORCH" in get_backends():
    import torch


    class TorchVisionClassificationDataset(AbstractDataset):
        """
        Wrapper for torchvision classification datasets to provide consistent API
        """

        def __init__(self, dataset, one_hot=False,
                     num_classes=None, **kwargs):
            """
            Parameters
            ----------
            dataset : :class:`torch.utils.data.Dataset`
                The actual dataset
            img_shape : tuple
                Height and width of output images (will be interpolated)
            **kwargs :
                Additional keyword arguments passed to the torchvision dataset
                class for initialization
            """
            super().__init__("", None, [], [])

            self.num_classes = None
            self.one_hot = one_hot
            self.data = dataset
            self.num_classes = num_classes

        def __getitem__(self, index):
            """
            return data sample specified by index
            Parameters
            ----------
            index : int
                index to specifiy which data sample to return
            Returns
            -------
            dict
                data sample
            """

            data = self.data[index]

            if isinstance(data[0], torch.Tensor):
                _data = data[0].numpy()

            else:
                _data = np.array(data[0])

            if isinstance(data[1], torch.Tensor):
                _label = data[1].numpy()
            else:
               _label = np.array(data[1])

            data_dict = {"data": _data,
                         "label": _label.reshape(1).astype(np.float32)}

            if self.one_hot:
                # TODO: Remove and refer to batchgenerators transform:
                # https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/transforms/utility_transforms.py#L97
                def make_onehot(num_classes, labels):
                    """
                    Function that converts label-encoding to one-hot format.
                    Parameters
                    ----------
                    num_classes : int
                        number of classes present in the dataset

                    labels : np.ndarray
                        labels in label-encoding format

                    Returns
                    -------
                    np.ndarray
                        labels in one-hot format
                    """
                    if isinstance(labels, list) or isinstance(labels, int):
                        labels = np.asarray(labels)
                    assert isinstance(labels, np.ndarray)
                    if len(labels.shape) > 1:
                        one_hot = np.zeros(shape=(list(labels.shape) + [num_classes]),
                                           dtype=labels.dtype)
                        for i, c in enumerate(np.arange(num_classes)):
                            one_hot[..., i][labels == c] = 1
                    else:
                        one_hot = np.zeros(shape=([num_classes]),
                                           dtype=labels.dtype)
                        for i, c in enumerate(np.arange(num_classes)):
                            if labels == c:
                                one_hot[i] = 1
                    return one_hot

                data_dict['label'] = make_onehot(self.num_classes, data_dict['label'])

            img = data_dict["data"]

            if len(img.shape) <= 3:
                img = img.reshape(
                    *img.shape, 1)

            img = img.transpose(
                (len(img.shape) - 1, *range(len(img.shape) - 1)))

            data_dict["data"] = img.astype(np.float32)
            return data_dict

        def __len__(self):
            """
            Return Number of samples
            Returns
            -------
            int
                number of samples
            """
            return len(self.data)

        def _make_dataset(self, path):
            raise NotImplementedError("This method is not needed by this class"
                                      " and should never be called")


    class TorchVisionMNIST(TorchVisionClassificationDataset):
        def __init__(self, train, root="/tmp/", download=True,
                     one_hot=False):
            from torchvision.datasets import MNIST

            dset = MNIST(root, train, download)
            super().__init__(dset, one_hot, 10)


    class TorchVisionEMNIST(TorchVisionClassificationDataset):
        def __init__(self, train, split, root="/tmp/", download=True,
                     one_hot=False):
            from torchvision.datasets import EMNIST

            dset = EMNIST(root, split, download=download, train=train)
            num_classes = {'byclass': 62, 'bymerge': 47, 'balanced': 47,
                           'letters': 26, 'digits': 10, 'mnist': 10}[split]

            super().__init__(dset, one_hot, num_classes)


    class TorchVisionFashionMNIST(TorchVisionClassificationDataset):
        def __init__(self, train, root="/tmp/", download=True, one_hot=False):
            from torchvision.datasets import FashionMNIST

            dset = FashionMNIST(root, train, download=download)
            super().__init__(dset, one_hot, 10)


    class TorchVisionKMNIST(TorchVisionClassificationDataset):
        def __init__(self, train, root="/tmp/", download=True, one_hot=False):
            from torchvision.datasets import KMNIST

            dset = KMNIST(root, train, download=download)
            super().__init__(dset, one_hot, 10)


    class TorchVisionFakeData(TorchVisionClassificationDataset):
        def __init__(self, size=1000, image_size=(3, 224, 224), num_classes=10, random_offset=0, one_hot=False):
            from torchvision.datasets import FakeData

            dset = FakeData(size=size, image_size=image_size,
                            num_classes=num_classes,
                            random_offset=random_offset)

            super().__init__(dset, one_hot,  num_classes)


    class TorchVisionLSUN(TorchVisionClassificationDataset):
        def __init__(self, classes, root, one_hot=False):

            from torchvision.datasets import LSUN
            dset = LSUN(root, classes)
            super().__init__(dset, one_hot, len(dset.classes))


    class TorchVisionImageFolder(TorchVisionClassificationDataset):
        def __init__(self, root, one_hot=False, load_fn=None):
            from torchvision.datasets import ImageFolder

            if load_fn is None:
                from torchvision.datasets.folder import default_loader
                load_fn = default_loader
            dset = ImageFolder(root, loader=load_fn)

            super().__init__(dset, one_hot, len(dset.classes))


    TorchVisionImageNet = TorchVisionImageFolder


    class TorchVisionDatasetFolder(TorchVisionClassificationDataset):
        def __init__(self, root, load_fn, extensions, one_hot):
            from torchvision.datasets import DatasetFolder

            dset = DatasetFolder(root, load_fn, extensions)

            super().__init__(dset, one_hot, len(dset.classes))


    class TorchVisionCIFAR10(TorchVisionClassificationDataset):
        def __init__(self, train, root="/tmp/", download=True, one_hot=False):
            from torchvision.datasets import CIFAR10

            dset = CIFAR10(root, train, download=download)
            super().__init__(dset, one_hot, len(dset.classes))


    class TorchVisionCIFAR100(TorchVisionClassificationDataset):
        def __init__(self, train, root="/tmp/", download=True, one_hot=False):
            from torchvision.datasets import CIFAR100

            dset = CIFAR100(root, train, download=download)
            super().__init__(dset, one_hot, len(dset.classes))


    class TorchVisionSTL10(TorchVisionClassificationDataset):
        def __init__(self, split, root="/tmp/", download=True, one_hot=False):
            from torchvision.datasets import STL10

            dset = STL10(root, split, download=download)
            super().__init__(dset, one_hot, len(dset.classes))


    class TorchVisionSVHN(TorchVisionClassificationDataset):
        def __init__(self, split, root="/tmp/", download=True, one_hot=False):
            from torchvision.datasets import SVHN

            dset = SVHN(root, split, download=download)
            super().__init__(dset, one_hot, 10)
