import os
from deliravision.data.base_datasets import ImageFolder, Downloadable
import tarfile
import gzip

IMAGENET_URLS = {
    "train": "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/"
             "ILSVRC2012_img_train.tar",
    "val": "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/"
           "ILSVRC2012_img_val.tar",
    "devkit": "http://www.image-net.org/challenges/LSVRC/2012/nnoupb/"
              "ILSVRC2012_devkit_t12.tar.gz"
}


class ImageNet(ImageFolder, Downloadable):
    def __init__(self, root="/tmp", train=True, download=True, remove=False):

        self.train = train

        Downloadable.__init__(self, path=root, download=download,
                              remove=remove)

        ImageFolder.__init__(self, path=root)

    @property
    def urls(self) -> dict:
        train_str = "train" if self.train else "val"
        return {IMAGENET_URLS[train_str]:
                    "ILSVRC2012_img_%s.tar" % train_str,
                IMAGENET_URLS["devkit"]: "ILSVRC2012_devkit_t12.tar.gz"}

    def preprocess_data(self, download_path, prep_path):
        for url, fname in self.urls.items():
            if "devkit" in fname:
                self._prepare_devkit(os.path.join(download_path, fname),
                                     prep_path)
            elif "train" in fname:
                self._prepare_train_split(os.path.join(download_path, fname),
                                          prep_path)
            else:
                self._prepare_val_split(os.path.join(download_path, fname),
                                        prep_path)

    @staticmethod
    def _prepare_devkit(file_path, prep_path):
        with gzip.GzipFile(file_path) as f_src:
            with open(file_path.replace(".gz", ""), "rb") as f_dst:
                f_dst.write(f_src.read())

        with tarfile.TarFile(file_path.replace(".gz", "")) as f:
            f.extractall(prep_path)

    @staticmethod
    def _prepare_train_split(file_path, prep_path):
        pass

    @staticmethod
    def _prepare_val_split(file_path, prep_path):
        pass
