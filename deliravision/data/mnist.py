from deliravision.data.base_datasets import ImageFolder, Downloadable
import codecs
import gzip
import os
import numpy as np
from PIL import Image


class MNIST(ImageFolder, Downloadable):
    """
    The MNIST Dataset

    See Also
    --------
    :class:`deliravision.data.base_datasets.ImageFolder`
        the Image Folder, this class is implemented upon.

    References
    ----------
    http://yann.lecun.com/exdb/mnist/

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
        root = os.path.join(root, "MNIST")

        if train:
            root = os.path.join(root, "train")
        else:
            root = os.path.join(root, "val")

        self.train = train

        Downloadable.__init__(self, path=root, download=download,
                              remove=remove)
        ImageFolder.__init__(self, path=os.path.join(root, "processed"))

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
        images = None
        labels = None

        # unzip files
        for file in self.urls.values():
            _, img_label, _ = file.split("-")
            if img_label == "images":
                images = self._read_images_from_binary_gzip(
                    os.path.join(download_path, file))
            else:
                labels = self._read_labels_from_binary_gzip(
                    os.path.join(download_path, file))

        # check if images and labels have been loaded
        assert images is not None and labels is not None

        # counter for each class
        label_idxs = {}
        for img, label in zip(images, labels):
            label = str(label)

            # set counter for class to zero and create dir for class if first
            # item of this class
            if label not in label_idxs:
                label_idxs[label] = 0
                os.makedirs(os.path.join(prep_path, label))

            # write image to disk and increase counter
            Image.fromarray(img).save(os.path.join(prep_path, label, "%05d.png"
                                                   % label_idxs[label]))
            label_idxs[label] += 1

    @staticmethod
    def _get_int_from_hex(b):
        """
        Helper function to decode an integer from a binary hexfile

        Parameters
        ----------
        b :
            binary buffer

        Returns
        -------
        int
            integer value decoded from given buffer

        """
        return int(codecs.encode(b, 'hex'), 16)

    def _read_images_from_binary_gzip(self, path):
        """
        Reads images from a binary gzip data file

        Parameters
        ----------
        path : str
            path to that datafile

        Returns
        -------
        :class:`numpy.ndarray`
            the loaded images

        """
        # open file and read data
        with gzip.open(path, "rb") as f:
            data = f.read()
        assert self._get_int_from_hex(data[:4]) == 2051

        # get number of items, rows and columns
        length = self._get_int_from_hex(data[4:8])
        num_rows = self._get_int_from_hex(data[8:12])
        num_cols = self._get_int_from_hex(data[12:16])

        # load from buffer with numpy
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return parsed.view(length, num_rows, num_cols)

    def _read_labels_from_binary_gzip(self, path):
        """
        Reads labels from a binary gzip data file

        Parameters
        ----------
        path : str
            path to that datafile

        Returns
        -------
        :class:`numpy.ndarray`
            the loaded labels

        """
        # open file and read data
        with gzip.open(path, 'rb') as f:
            data = f.read()
        assert self._get_int_from_hex(data[:4]) == 2049

        # get number of items
        length = self._get_int_from_hex(data[4:8])

        # read from buffer with numpy
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return parsed.view(length)

    @property
    def urls(self):
        """
        Property returning the urls of the current mode

        Returns
        -------
        dict
            dictionary containing either the train or test urls

        """
        if self.train:
            return {
                "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz":
                    "train-images-idx3-ubyte.gz",
                "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz":
                    "train-labels-idx1-ubyte.gz"
            }

        return {
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz":
                "t10k-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz":
                "t10k-labels-idx1-ubyte.gz"
        }
