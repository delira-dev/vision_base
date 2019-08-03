import os
import tarfile
import gzip
import numpy as np
from PIL import Image

from deliravision.data.base_datasets import ImageFolder, Downloadable


class STL10(ImageFolder, Downloadable):
    """
    The STL10 Dataset

    References
    ----------
    https://cs.stanford.edu/~acoates/stl10/
    """

    def __init__(self, root="/tmp", split="train", download=True,
                 remove=False):
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

        assert split in ("train", "unlabeled", "test")
        self.split = split

        root = os.path.join(root, self.name)

        Downloadable.__init__(self, path=root, download=download,
                              remove=remove)

        ImageFolder.__init__(self, os.path.join(root, split))

    @property
    def name(self):
        """
        Property returning the Datasets name

        Returns
        -------
        str
            the name

        """

        return "STL10"

    @property
    def urls(self):
        """
        Property returning the url

        Returns
        -------
        dict
            dictionary containing the url and the filename

        """
        return {
            "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz":
                "stl10_binary.tar.gz"
        }

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

        # unpack gzip file
        with gzip.open(os.path.join(download_path, fname), "rb") as f_src:
            with open(os.path.join(download_path,
                                   fname.replace(".gz", "")), "wb") as f_dst:
                f_dst.write(f_src.read())

        # unpack tar file
        with tarfile.TarFile(os.path.join(download_path,
                                          fname.replace(".gz", ""))) as f:
            f.extractall(download_path)

        download_path = os.path.join(download_path, fname.replace(".tar.gz",
                                                                  ""))

        images, labels = None, None
        img_file = os.path.join(download_path, "%s_X.bin" % self.split)
        label_file = os.path.join(download_path, "%s_y.bin" % self.split)

        # load images
        if os.path.isfile(img_file):
            with open(img_file, "rb") as f:
                images = np.fromfile(f, dtype=np.uint8).reshape(-1, 96, 96, 3)

        # load labels
        if os.path.isfile(label_file):
            with open(label_file, "rb") as f:
                # labels are stored in matlab format starting with 1
                labels = np.fromfile(f, dtype=np.uint8) - 1

        # load class names
        with open(os.path.join(download_path, "class_names.txt"), "r") as f:
            class_names = f.readlines()

        if labels is None:
            labels = tuple([None] * len(images))

        prep_path = os.path.join(prep_path, self.split)

        class_ctr = {}
        # save labels to directories based on class nabes
        for img, label in zip(images, labels):
            if label is None:
                label = "unknown"
            else:
                label = class_names[label]

            if label not in class_ctr:
                class_ctr[label] = 0
            os.makedirs(os.path.join(prep_path, label))
            Image.fromarray(img).save(os.path.join(prep_path, label,
                                                   "%06d.png"
                                                   % class_ctr[label]))

            class_ctr[label] += 1

    def __getitem__(self, item):
        """
        Overwrites the ImageFolder's ``__getitem__`` to avoid returning a
        label for the unlabeled cases (which would be done otherwise since the
        files exist in a subdirectory, which would result in a 1-class dataset)

        Parameters
        ----------
        item : int

        Returns
        -------
        dict
            the returned sample

        """
        sample = ImageFolder.__getitem__(self, item)
        if self.split == "unlabeled":
            # remove label since it is unknown
            sample.pop("label")

        return sample
