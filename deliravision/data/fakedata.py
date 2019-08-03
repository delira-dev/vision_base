from deliravision.data.base_datasets import ClassificationDataset
import numpy as np


class ClassificationFakeData(ClassificationDataset):
    """
    Dataset generating consistent random fake data for classification purposes
    """

    def __init__(self, num_samples=1000, img_size=(3, 224, 224),
                 num_classes=10, rng_offset=0):
        """

        Parameters
        ----------
        num_samples : int
            number of samples in the dataset
        img_size : tuple
            shape of a single image (with channels at front)
        num_classes : int
            number of classes
        rng_offset : int
            The random number generator is set for each image generation with
            the given index and this optional offset
        """

        super().__init__(None, None)

        self.num_samples = num_samples
        self._num_classes = num_classes
        self.img_size = img_size
        self.rng_offset = rng_offset

    def __getitem__(self, index):
        """
        Returns the sample specified by the given index
        Parameters
        ----------
        index

        Returns
        -------
        dict
            the dictionary containing the datatype

        """

        # save previous random state
        rng_state = np.random.get_state()
        # seed random generator with index and offset
        np.random.seed(self.rng_offset + index)

        try:
            return {
                "data": np.random.randn(*self.img_size),
                "label": np.random.randint(0, self.num_classes, (1, ))
            }
        finally:
            # reset random state
            np.random.set_state(rng_state)

    @property
    def num_classes(self) -> int:
        """
        Number of classes

        Returns
        -------
        int
            number of classes
        """
        return self._num_classes

    def _make_dataset(self, path: str):
        pass

    def __len__(self):
        """
        Length of the dataset

        Returns
        -------
        int
            length of the dataset

        """
        return self.num_samples
