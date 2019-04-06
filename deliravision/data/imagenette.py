from delira import get_backends

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

if "TORCH" in get_backends():

    from .torchvision_classification import TorchVisionImageFolder
    from torchvision.datasets.utils import download_url
    import tarfile
    import os

    def process_url(url, dst, name, remove_file=True):
        _target_dir = os.path.join(dst, name.rsplit(".", 1)[0])

        # download file if dataset does not exist
        if not (os.path.isdir(_target_dir) or
                os.path.isfile(os.path.join(dst, name))):
            download_url(url, dst, name)

        # extract file if dir does not exist
        if not os.path.isdir(_target_dir):

            with tarfile.open(os.path.join(dst, name)) as f:
                _target_dir = os.path.join(dst, name.rsplit(".")[0])
                os.makedirs(_target_dir, exist_ok=True)
                f.extractall(_target_dir)

        # remove file if flag specified and file exists
        if remove_file and os.path.isfile(os.path.join(dst, name)):
            os.remove(os.path.join(dst, name))

        return os.path.join(_target_dir, os.listdir(_target_dir)[0])


    class ImageNette(TorchVisionImageFolder):
        def __init__(self, split, size="full", root="/tmp/", one_hot=False,
                     load_fn=None, remove=True):

            dset_dir = process_url(IMAGENETTE_URLS[size], root,
                                   "ImageNette_%s.tgz" % size,
                                   remove_file=remove)

            data_dir = os.path.join(dset_dir, split)

            super().__init__(data_dir, one_hot=one_hot, load_fn=load_fn)


    class ImageWoof(TorchVisionImageFolder):
        def __init__(self, split, size="full", root="/tmp/", one_hot=False,
                     load_fn=None, remove=True):

            dset_dir = process_url(IMAGENETTE_URLS[size], root,
                                   "ImageWoof_%s.tgz" % size,
                                   remove_file=remove)

            data_dir = os.path.join(dset_dir, split)

            super().__init__(data_dir, one_hot=one_hot, load_fn=load_fn)
