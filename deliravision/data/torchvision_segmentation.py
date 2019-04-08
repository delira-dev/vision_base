from delira import get_backends

if "TORCH" in get_backends:
    import numpy as np
    import torch

    from delira.data_loading import AbstractDataset

    class TorchVisionSegmentationDataset(AbstractDataset):
        """
        Abstractt Dataset to wrap torchvision segmentation datasets
        """

        def __init__(self, dset, seg_key="label", **kwargs):
            """

            Parameters
            ----------
            dset : Any
                the dataset to wrap
            seg_key : str
                key holding the segmentation in returned data dicts
                (default: "label")
            **kwargs :
                additional keyword arguments

            """
            super().__init__(None, None, None, None)

            self.data = dset
            self.seg_key = seg_key

        def __getitem__(self, index):
            """
            Returns a single sample from the dataset

            Parameters
            ----------
            index : int
                the index of the sample to return

            Returns
            -------
            dict
                returned sample

            """

            img, seg = self.get_sample_from_index(index)

            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()

            else:
                img = np.array(img)

            if isinstance(seg, torch.Tensor):
                seg = seg.detach().cpu().numpy()
            else:
                seg = np.array(seg)

            img = img.astype(np.float)
            seg = seg.astype(np.float32)

            if img.ndim < 3:
                img = img.reshape((*img.shape, 1))

            if seg.ndim < 3:
                seg = seg.reshape((*seg.shape, 1))

            img = img.transpose(2, 0, 1)
            seg = seg.transpose(2, 0, 1)

            return {
                "data": img,
                self.seg_key: seg
            }

        def __len__(self):
            """
            Returns the dataset's length

            Returns
            -------
            int
                dataset's length

            """
            return len(self.data)

    class TorchVisionVOCSegmentation(TorchVisionSegmentationDataset):
        def __init__(self, image_set, root="/tmp/", year='2012', download=False,
                     seg_key="label"):

            from torchvision.datasets import VOCSegmentation

            dset = VOCSegmentation(root=root, year=year, image_set=image_set,
                                   download=download)

            super().__init__(dset=dset, seg_key=seg_key)


    class TorchVisionCityScapes(TorchVisionSegmentationDataset):
        def __init__(self, split, root="/tmp/", mode='fine',
                     target_type='instance', seg_key="label"):

            from torchvision.datasets import Cityscapes

            dset = Cityscapes(root=root, split=split, mode=mode,
                              target_type=target_type)

            super().__init__(dset=dset, seg_key=seg_key)



