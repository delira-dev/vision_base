import unittest
import numpy as np


class TensorOpTest(unittest.TestCase):
    def setUp(self) -> None:
        self._labels = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0])
        self._n_classes = 6
        self._targets = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
        ], dtype=np.float)

    def test_onehot_npy(self):
        from deliravision.utils.tensor_ops import make_onehot_npy

        self.assertListEqual(self._targets.tolist(),
                             make_onehot_npy(self._labels,
                                             self._n_classes).tolist())


if __name__ == '__main__':
    unittest.main()