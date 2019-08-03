from deliravision.data.base_datasets import ImageFolder, Downloadable
import os
import gzip
import tarfile

IMAGENETTE_URLS = {
    "full": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette.tgz",
    "320px": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette-320.tgz",
    "160px": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz"
}

IMAGEWOOF_URLS = {
    "full": "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof.tgz",
    "320px": "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof-320.tgz",
    "160px": "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof-160.tgz"
}


class ImageNette(ImageFolder, Downloadable):
    """
    The Imagenette Dataset is a 10-class imagenet-subset of relatively easy
    distinguishable classes

    See Also
    --------
    :class:`deliravision.data.base_datasets.ImageFolder`
        the Image Folder, this class is implemented upon.
    :class:`deliravision.data.imagenette.ImageWoof`
        a harder 10-class imagenet subset

    References
    ----------
    https://github.com/fastai/imagenette

    """

    def __init__(self, root="/tmp", resolution="full", train=True,
                 download=True, remove=False):
        """

        Parameters
        ----------
        root : str
            the path, all data should be placed in;
            will be created if not yet existing
        resolution : str
            a string specifying the subset to load, this can be one of the
            following: 'full' | '320px' | '160px'
        train : bool
            whether to load the trainset or the testset
        download : bool
            whether to download the dataset; This will be only done, if it
            wasn't downlaoded already
        remove : bool
            whether to remove the downlaoded data after processing it
        """
        assert resolution in ("full", "320px", "160px")
        self.resolution = resolution
        root = os.path.join(root, self.name, str(self.resolution))

        Downloadable.__init__(self, path=root, download=download,
                              remove=remove)

        root = os.path.join(root, "processed", self.name.lower())

        if train:
            root = os.path.join(root, "train")
        else:
            root = os.path.join(root, "val")

        ImageFolder.__init__(self, path=root)

    @property
    def name(self):
        """
        Property to specify the datasets name

        Returns
        -------
        str
            the name
        """
        return "ImageNette"

    @property
    def urls(self) -> dict:
        """
        Property returning the URLs to download the data if necessary based on
        the specified resolution

        Returns
        -------
        dict
            a combination of URLs and filenames
        """
        return {IMAGENETTE_URLS[self.resolution]:
                "imagenette_%s.tgz" % self.resolution}

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
        with gzip.open(os.path.join(download_path, fname)) as gfile:
            with open(os.path.join(download_path,
                                   fname.replace(".tgz", ".tar")), "wb") as f:
                f.write(gfile.read())

        with tarfile.TarFile(os.path.join(download_path,
                                          fname.replace(".tgz", ".tar"))
                             ) as tfile:
            tfile.extractall(prep_path)


class ImageWoof(ImageNette):
    """
    The ImageWoof Dataset is a 10-class imagenet-subset of relatively hard +
    (compared to Imagenette) distinguishable classes

    See Also
    --------
    :class:`deliravision.data.base_datasets.ImageFolder`
        the Image Folder, this class is implemented upon.
    :class:`deliravision.data.imagenette.ImageNette`
        a harder 10-class imagenet subset

    References
    ----------
    https://github.com/fastai/imagenette

    """

    @property
    def name(self):
        """
        Property to specify the datasets name

        Returns
        -------
        str
            the name
        """
        return "ImageWoof"

    @property
    def urls(self) -> dict:
        """
        Property returning the URLs to download the data if necessary based on
        the specified resolution

        Returns
        -------
        dict
            a combination of URLs and filenames
        """
        return {IMAGEWOOF_URLS[self.resolution]:
                "imagewoof_%s.tgz" % self.resolution}
