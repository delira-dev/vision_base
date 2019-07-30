from delira.data_loading import AbstractDataset
from abc import abstractmethod
import os
from PIL import Image
import numpy as np
import SimpleITK as sitk
import wget
import shutil
from multiprocessing import Pool
from functools import partial

from deliravision.data.utils import get_files_from_dir


class ClassificationDataset(AbstractDataset):
    """
    API definintion for Classification Datasets
    """
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """
        Property returning the number of classes

        Returns
        -------
        int
            the number of classes

        """
        raise NotImplementedError


class Downloadable:
    """
    API definition and download code for downloadable items
    """
    def __init__(self, path, download=True, remove=False):
        """

        Parameters
        ----------
        path : str
            the root path which will contain the items later on
        download : bool
            specifies whether to download the items if not already done
        remove : bool
            specifies whether to remove the items after processing them
        """

        prep_path = os.path.join(path, "preprocessed")
        # check if directory exists and is not empty
        if not self.check_for_path(prep_path):

            # if not preprocessed already: check if data has to be downloaded
            download_path = os.path.join(path, "raw")
            if download and not self.check_for_path(download_path):

                # if necessary: download data
                os.makedirs(download_path)
                for url, fname in self.urls.items():
                    wget.download(url, os.path.join(download_path, fname))

            # preprocess data
            self.preprocess_data(download_path, prep_path)

            # remove data if wanted
            if remove:
                shutil.rmtree(download_path, ignore_errors=True)

    @property
    @abstractmethod
    def urls(self) -> dict:
        """
        Abstract Property to get the items url for downloading the data

        Returns
        -------
        dict
            dictionary with urls as keys and filenames as value

        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @staticmethod
    def check_for_path(path):
        """
        Helper function to check, whether a path is a non-empty directory
        Parameters
        ----------
        path : str
            the path to check for

        Returns
        -------
        bool
            whether the path is a non-empty directory

        """
        return os.path.isdir(path) and any(os.listdir(path))


class DataFolder(ClassificationDataset):
    """
    A datafolder loading items of arbitrary type with a specific load function.
    The given path must contain subdirs and the items in each subdir will be
    considered as items of one class.
    """
    def __init__(self, path, load_fn, extensions: tuple):
        """

        Parameters
        ----------
        path : str
            the root path; must contain a subdirectory with items for each
            class
        load_fn :
            function to load specific items to numpy arrays
        extensions : tuple
            a tuple of strings to check the images validity
        """
        super().__init__(data_path=path, load_fn=load_fn)

        self._extensions = extensions
        self._class_mappings, self._data = self._make_dataset(path)

    def _make_dataset(self, path: str):
        """
        Function to return the class mappings and the files in the
        subdirectories

        Parameters
        ----------
        path : str
            the root path; must contain a subdirectory with items for each
            class

        Returns
        -------
        dict
            the class mappings from directories to index-based classes
        tuple
            the files which are part of the dataset

        """
        class_mappings = {}
        files = []
        for item in sorted(os.listdir(path)):
            whole_path = os.path.join(path, item)
            # check if item is directory
            if os.path.isdir(whole_path):
                # parse files from dir
                files += get_files_from_dir(whole_path,
                                            self._extensions)

                # add new mapping for current item
                class_mappings[item] = len(class_mappings)

        return self._class_mappings, tuple(files)

    @property
    def class_mappings(self):
        """
        Property to return a class mapping from the indices to the real class
        names (specified by the names of the folders the items are stored in)

        Returns
        -------
        dict
            class mapping from index to name

        """
        # revert keys and values, because this should enable a remapping from
        # indices to names
        return {v: k for k, v in self._class_mappings.items()}

    @property
    def num_classes(self):
        """
        Property to return the number of classes

        Returns
        -------
        int
            the number of classes

        """
        return len(self._class_mappings)

    def __getitem__(self, index):
        """
        Makes the class indexeable and returns the sample loaded by the load
        function

        Parameters
        ----------
        index : int
            the index specifying which index to load

        Returns
        -------
        dict
            dictionary containing the current sample

        """
        data_path = self.data[index]
        return {"data": data_path,
                "label": self._class_mappings[os.path.basename(data_path)]
                }

    def get_sample_from_index(self, index):
        """
        Function to specify the mapping of the index to the sample

        Parameters
        ----------
        index : int
            the index specifying which sample to load

        Returns
        -------
        dict
            the dictionary containing the actual sample

        """
        return self.__getitem__(index)

    def __len__(self):
        """
        Function to define the datasets length

        Returns
        -------
        int
            number of samples in this dataset

        """
        return len(self._data)


def pil_loader(path):
    """
    Function to load a single image

    Parameters
    ----------
    path : str
        path specifying which image to load

    Returns
    -------
    :class:`numpy.ndarray`
        the loaded image

    """
    with open(path, 'rb') as f:
        img = Image.open(f)
    img = np.array(img)

    # add channel dimension if necessary
    if len(img.shape) == 2:
        img = img[None, ...]
    else:
        # move channels to front
        img = np.moveaxis(img, -1, 0)
    return img


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


class ImageFolder(DataFolder):
    """
    ImageFolder Class loading the most common image types with a default loader

    See Also
    --------
    :class:`DataFolder`
        a generalized version of this class. Can be parametrized to achieve
        other behaviors if needed
    :class:`MedicalImageFolder`
        image folder loading the most common medical image types
    """
    def __init__(self, path):
        """

        Parameters
        ----------
        path : str
            the root path; must contain a subdirectory with items for each
            class
        """
        super().__init__(path=path, load_fn=pil_loader,
                         extensions=IMG_EXTENSIONS)


MED_EXTENSIONS = ('.mnc', '.mrc', '.mha', '.mhd', '.nia', '.nii.gz', '.hdr',
                  '.img', '.img.gz', '.nrrd', '.nhdr', '.vtk', '.pic', '.gipl')


def sitk_loader(path):
    """
    Loader function to load medical images with SimpleITK

    Parameters
    ----------
    path : str
        path specifying which image to load

    Returns
    -------
    :class:`numpy.ndarray`
        the loaded image

    """
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


class MedicalImageFolder(DataFolder):
    """
    An ImageFolder to load the most common medical image types with SimpleITK.

    See Also
    --------
    :class:`DataFolder`
        a generalized version of this class. Can be parametrized to achieve
        other behaviors if needed
    :class:`ImageFolder`
        image folder loading the most common non-medical image types
    """
    def __init__(self, path):
        """

        Parameters
        ----------
        path : str
            the root path; must contain a subdirectory with items for each
            class
        """
        super().__init__(path=path, load_fn=sitk_loader,
                         extensions=MED_EXTENSIONS)


def _get_data_by_indexing(dataset, index):
    return dataset[index]


class Lazy2Cache(AbstractDataset):
    """
    Wraps a lazy dataset, loads all the data (in a parallel manner if wanted)
    and caches it

    Warnings
    --------
    For large datasets, this may take a while.
    Also your machine may run out of memory or become extremely slow.
    """
    def __init__(self, dataset, n_jobs=None):
        """

        Parameters
        ----------
        dataset : :class:`delira.data_loading.AbstractDataset`
        n_jobs : int
            number of jobs to start for loading
            if 0: only main process will be used
            if int > 0: specifies the number of processes
            if None: One process per CPU will be used
        """
        super().__init__(None, None)

        self._n_jobs = n_jobs
        self.data = self._make_dataset(dataset)

    def _make_dataset(self, dataset):
        """
        Helper function to load all data

        Parameters
        ----------
        dataset : :class:`delira.data_loading.AbstractDataset`
            the lazy dataset

        Returns
        -------
        tuple
            the loaded samples

        """
        load_fn = partial(_get_data_by_indexing, dataset=dataset)
        if self._n_jobs is None or self._n_jobs >= 1:
            with Pool(self._n_jobs) as p:
                data = p.map(load_fn, range(len(dataset)))

        else:
            data = map(load_fn, range(len(dataset)))

        return tuple(data)

    def __getitem__(self, item):
        """
        Defines what to do during indexing

        Parameters
        ----------
        item : int
            index specifying the sample to return

        Returns
        -------
        dict
            dictionary containing the sample

        """
        return self.data[item]

    def get_sample_from_index(self, index):
        """
        Proxy mapping the index to a sample

        Parameters
        ----------
        index : int
            the index specifying the sample to return

        Returns
        -------
        dict
            the specified sample
        """
        return self.__getitem__(index)

    def __len__(self):
        """
        Returns the dataset's length

        Returns
        -------
        int
            number of samples in this dataset

        """
        return len(self.data)