from deliravision.data.base_datasets import Downloadable, ImageFolder
import gzip
import tarfile
import os
import pickle
from PIL import Image


class _CifarDataset(ImageFolder, Downloadable):
    """
    Basic Dataset for the CIFAR10 and CIFAR100 Datasets
    """
    def __init__(self, cifar_type, root="/tmp", train=True, download=True,
                 remove=False):
        """

        Parameters
        ----------
        cifar_type : str
            string indicating which cifar dataset to load;
            Should be one of '10' | '100'
        root : str
            the path, all data should be placed in;
            will be created if not yet existing
        train : bool
            whether to load the trainset or the testset
        download : bool
            whether to download the dataset; This will be only done, if it
            wasn't downlaoded already
        remove : bool
            whether to remove the downlaoded data after processing it
        """

        # conversion in case an int or float was passed
        cifar_type = str(cifar_type)
        assert cifar_type in ("10", "100")
        self.train = train
        self._cifar_type = cifar_type

        root = os.path.join(root, self.name)
        Downloadable.__init__(self, path=root, download=download,
                              remove=remove)

        ImageFolder.__init__(self,
                             os.path.join(root, "train" if train else "test"))

    @property
    def name(self):
        """
        Property returning the Datasets name

        Returns
        -------
        str
            the name

        """
        return "CIFAR%s" % self._cifar_type

    @property
    def urls(self) -> dict:
        """
        Property returning the urls

        Returns
        -------
        dict
            dictionary containing either the cifar10 or the cifar100 urls

        """
        return {"https://www.cs.toronto.edu/~kriz/cifar-%s-python.tar.gz"
                % self._cifar_type:
                    "cifar-%s-python.tar.gz" % self._cifar_type}

    def preprocess_data(self, download_path, prep_path):
        """
        Function to preprocess the downloaded data

        Parameters
        ----------
        download_path : str
            the path containing the downloaded data
        prep_path : str
            the path the preprocessed data should be stored in

        """
        fname = list(self.urls.values())[0]

        # extract gzip file
        with gzip.GzipFile(os.path.join(download_path, fname)) as f_src:
            with open(os.path.join(download_path,
                                   fname.replace(".gz", "")), "wb") as f_dst:
                f_dst.write(f_src.read())

        # extract tarfile
        with tarfile.TarFile(os.path.join(download_path,
                                          fname.replace(".gz", ""))) as f:
            f.extractall(download_path)

        download_path = os.path.join(download_path, "cifar-%s-batches-py"
                                     % self._cifar_type)

        # load class names
        with open(os.path.join(prep_path, "batches.meta"), "rb") as f:
            class_names = pickle.load(f, encoding="latin1")["label_names"]

        if self.train:
            data_files = ("data_batch_%d" % i for i in range(1, 6))
            prep_path = os.path.join(prep_path, "train")
        else:
            data_files = ("test_batch",)
            prep_path = os.path.join(prep_path, "test")

        # load images and labels and write them to folders
        for file in data_files:
            with open(os.path.join(download_path, file), "rb") as f:
                data_dict = pickle.load(f, encoding="latin1")

            for data, label, filename in zip(data_dict["data"],
                                             data_dict["labels"],
                                             data_dict["filenames"]):
                label = class_names[label]
                os.makedirs(os.path.join(prep_path, str(label)),
                            exist_ok=True)
                Image.fromarray(data.reshape(32, 32, 3)).save(
                    os.path.join(prep_path, str(label), filename)
                )


class CIFAR10(_CifarDataset):
    """
    CIFAR10 Dataset

    See Also
    --------
    :class:`deliravision.data.cifar.CIFAR100`
        the CIFAR100 dataset
    :class:`deliravision.data.cifar._CifarDataset`
        the internal implementation of generic CIFAR datasets

    References
    ----------
    http://www.cs.toronto.edu/~kriz/cifar.html

    """
    def __init__(self, root="/tmp", train=True, download=True, remove=False):
        """

        Parameters
        ----------
        root : str
            the path, all data should be placed in;
            will be created if not yet existing
        train : bool
            whether to load the trainset or the testset
        download : bool
            whether to download the dataset; This will be only done, if it
            wasn't downlaoded already
        remove : bool
            whether to remove the downlaoded data after processing it
        """
        super().__init__("10", root=root, train=train, download=download,
                         remove=remove)


class CIFAR100(_CifarDataset):
    """
    CIFAR100 Dataset

    See Also
    --------
    :class:`deliravision.data.cifar.CIFAR10`
        the CIFAR10 dataset
    :class:`deliravision.data.cifar._CifarDataset`
        the internal implementation of generic CIFAR datasets

    References
    ----------
    http://www.cs.toronto.edu/~kriz/cifar.html

    """
    def __init__(self, root="/tmp", train=True, download=True, remove=False):
        """

        Parameters
        ----------
        root : str
            the path, all data should be placed in;
            will be created if not yet existing
        train : bool
            whether to load the trainset or the testset
        download : bool
            whether to download the dataset; This will be only done, if it
            wasn't downlaoded already
        remove : bool
            whether to remove the downlaoded data after processing it
        """
        super().__init__("100", root=root, train=train, download=download,
                         remove=remove)
