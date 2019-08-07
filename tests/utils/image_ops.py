import unittest
import numpy as np
import SimpleITK as sitk
from deliravision.utils.image_ops import bounding_box, \
    calculate_origin_offset, max_energy_slice, \
    sitk_copy_metadata, sitk_new_blank_image, \
    sitk_resample_to_image, sitk_resample_to_shape, sitk_resample_to_spacing


class ImageOpTest(unittest.TestCase):
    def setUp(self) -> None:
        self._img = np.zeros((45, 45, 45))
        self._img[23:25, 5:40, 10:35] = 1

        # convert array to nested list
        self._img = self._img.tolist()

    def test_bounding_box(self):

        bbox_list = bounding_box(self._img)
        self.assertTupleEqual(bbox_list, (23, 24, 5, 39, 10, 34))

        img_npy = np.array(self._img)
        bbox_npy = bounding_box(img_npy)
        self.assertTupleEqual(bbox_npy, (23, 24, 5, 39, 10, 34))

        img_sitk = sitk.GetImageFromArray(img_npy)
        bbox_sitk = bounding_box(img_sitk)
        self.assertTupleEqual(bbox_sitk, (23, 24, 5, 39, 10, 34))

    def test_calculate_origin_offset(self):
        offset = calculate_origin_offset((3., 1., 1.), (1., 1., 1.))
        # check for almost equal due to machine precision issues
        for _offset, _target_offset in zip(offset.tolist(), [1., 0., 0.]):
            self.assertAlmostEqual(_offset, _target_offset)

        offset = calculate_origin_offset((1.5, 1.4, 1.3), (1., 1., 1.))
        # check for almost equal due to machine precision issues
        for _offset, _target_offset in zip(offset.tolist(), [0.25, 0.2, 0.15]):
            self.assertAlmostEqual(_offset, _target_offset)

    def test_max_energy_slice(self):
        slice_idx = max_energy_slice(
            sitk.GetImageFromArray(
                np.array(
                    self._img)))
        self.assertIn(slice_idx, [23, 24])

    def test_copy_metadata(self):
        img_sitk = sitk.GetImageFromArray(np.array(self._img))
        img_sitk.SetMetaData("Foo", "Bar")

        blank_image = sitk.GetImageFromArray(np.zeros((10, 10, 10)))

        blank_image = sitk_copy_metadata(img_sitk, blank_image)

        for key in blank_image.GetMetaDataKeys():
            self.assertEqual(img_sitk.GetMetaData(key),
                             blank_image.GetMetaData(key))

    def test_new_blank_image(self):
        shape = (15, 13, 12)
        spacing = (1., 2., 3.)
        direction = (1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0)
        origin = (0., 0., 0.)
        img = sitk_new_blank_image(shape, spacing, direction, origin)

        self.assertTupleEqual(img.GetSize(), shape)
        self.assertTupleEqual(img.GetSpacing(), spacing)
        self.assertTupleEqual(img.GetDirection(), direction)
        self.assertTupleEqual(img.GetOrigin(), origin)

    def test_resample_to_image(self):
        src_img = sitk_new_blank_image((34, 34, 34), (2, 3, 4))

        resampled_img = sitk_resample_to_image(
            src_img, sitk.GetImageFromArray(np.array(self._img)))

        self.assertTupleEqual(
            resampled_img.GetSize(), np.array(
                self._img).shape)
        self.assertTupleEqual(resampled_img.GetSpacing(), (1., 1., 1.))

    def test_resample_to_shape(self):
        src_img = sitk_new_blank_image((34, 34, 34), (2, 3, 4))
        resampled_img = sitk_resample_to_shape(
            src_img, np.array(self._img).shape)

        self.assertTupleEqual(resampled_img.GetSize(),
                              np.array(self._img).shape)

    def test_resample_to_spacing(self):
        src_img = sitk.GetImageFromArray(np.array(self._img))

        resampled_img = sitk_resample_to_spacing(src_img, (2., 3., 4.))
        self.assertTupleEqual(resampled_img.GetSpacing(), (2., 3., 4.))


if __name__ == "__main__":
    unittest.main()
