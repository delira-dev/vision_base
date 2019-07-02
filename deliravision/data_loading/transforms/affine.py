from delira.data_loading.transforms.abstract_transform import BaseTransform
from scipy.linalg import inv
from scipy.ndimage import affine_transform
import numpy as np


class Affine(BaseTransform):
    """
    Affine Transformation for 1D, 2D and 3D data to be applied to images.

    Notes
    -----
    All arguments, which are passed on a per-dimension basis, must be given
    w.r.t the following order: ([z,] y,) x

    See Also
    --------
    :func:`scipy.ndimage.affine_transform`

    """
    def __init__(self, matrix=None, scale=None, translation=None,
                 rotation=None, source_keys=None, destination_keys=None,
                 n_dim=None, **kwargs):

        """

        Parameters
        ----------
        matrix : :class:`np.ndarray`
            the transformation matrix. If this argument is set, :param:`scale`,
            :param:`translation` and :param`rotation` are ignored
        scale : :class:`np.ndarray` or int or float
            the scale factors for each dimension; If only a single value was
            passed, this value will be used for each dimension;
            will be ignored, if :param:`matrix` was given
        translation : :class:`np.ndarray` or int or float
            the translation values for each dimension;
            If only a single value was  passed, this value will be used for
            each dimension;  will be ignored, if :param:`matrix` was given
        rotation : :class:`np.ndarray` or int or float
            the rotation factors for each dimension (in rad);
            If only a single value was  passed, this value will be used for
            each rotation axis; will be ignored, if :param:`matrix` was given
        source_keys : tuple or string
            the keys of the ``data_dict``, this trafo should be applied to
        destination_keys : tuple or string
            the keys of the ``data_dict``, this trafo's result should be saved
            to. If None: defaults to :param:`source_keys`
        n_dim : int
            dimensionality; if not given, an attempt to extract the
            dimensionality from the given data will be made
        **kwargs :
            additional keyword arguments (will be passed to
            :func:`scipy.ndimage.affine_transform`

        """

        if source_keys is None:
            source_keys = ("data",)
        if destination_keys is None:
            destination_keys = source_keys
        super().__init__(source_keys, destination_keys)
        if matrix is not None:
            matrix = inv(matrix)
        self._matrix = matrix

        self._scale = scale
        self._translation = translation
        self._rotation = rotation
        self._kwargs = kwargs
        self._n_dim = n_dim

    @staticmethod
    def _set_default_args(scale, translation, rotation, n_dim):
        """
        sets the default arguments if parameters are None or only scalar values.

        Parameters
        ----------
        scale :
            the scale factors
        translation :
            the translation values
        rotation :
            the rotation angles (in rad)
        n_dim : int
            the dimensionality

        Returns
        -------
        Iterable
            scale factors (1 per dimension)
        Iterable
            translation values (1 per dimension)
        Iterable
            rotation angles (in rad; 1 per rotation axis)

        """
        if scale is None:
            scale = 1

        if translation is None:
            translation = 0

        if rotation is None:
            rotation = 0

        if not isinstance(scale, (tuple, list, np.ndarray)):
            scale = [scale] * n_dim
        else:
            assert len(scale) == n_dim

        if not isinstance(translation, (tuple, list, np.ndarray)):
            translation = [translation] * n_dim
        else:
            assert len(translation) == n_dim

        n_rot_dim = n_dim - int(n_dim == 2)
        if not isinstance(rotation, (tuple, list, np.ndarray)):
            rotation = [rotation] * n_rot_dim
        else:
            assert len(rotation) == n_rot_dim

        return scale, translation, rotation

    @staticmethod
    def _ensemble_trafo_matrix(scale, translation, rotation, n_dim):
        """
        Ensembles the Actual transformation matrix

        Parameters
        ----------
        scale : Iterable
            the scale factors
        translation : Iterable
            the translation factors
        rotation : Iterable
            the rotation angles (in rad)
        n_dim : int
            the number of dimensions

        Returns
        -------
        :class:`np.ndarray`
            the transformation matrix

        """
        if n_dim == 1:
            # only consists of scale and translation for 1 d
            matrix = np.array([[scale[0], translation[0]],
                              [0, 1]])
        elif n_dim == 2:
            # only 1 rotation axis in 2d
            matrix = np.array([
                [np.cos(rotation[0]) * scale[1], -np.sin(rotation[0]),
                 translation[1]],
                [np.sin(rotation[0]), np.cos(rotation[0]) * scale[0],
                 translation[0]]
            ])

        elif n_dim == 3:
            # rotation around x-axis
            rot_x = np.eye(n_dim + 1)
            rot_x[1, 1] = np.cos(rotation[2])
            rot_x[1, 2] = -np.sin(rotation[2])
            rot_x[2, 1] = np.sin(rotation[2])
            rot_x[2, 2] = np.cos(rotation[2])

            # rotation around y-axis
            rot_y = np.eye(n_dim + 1)
            rot_y[0, 0] = np.cos(rotation[1])
            rot_y[0, 2] = np.sin(rotation[1])
            rot_y[2, 0] = - np.sin(rotation[1])
            rot_y[2, 2] = np.cos(rotation[1])

            # rotation around z-axis
            rot_z = np.eye(n_dim + 1)
            rot_z[0, 0] = np.cos(rotation[0])
            rot_z[0, 1] = -np.sin(rotation[0])
            rot_z[1, 0] = np.sin(rotation[0])
            rot_z[1, 1] = np.cos(rotation[0])

            # calculate total rotation matrix
            rotation = np.dot(rot_z, np.dot(rot_y, rot_x))

            # scale matrix (reversed order to transform z-y-x to x-y-z)
            scale = np.diag(scale[::-1])

            # translation matrix (reversed order to transform z-y-x to x-y-z)
            transl = np.eye(n_dim+1)
            transl[-1, :-1] = translation[::-1]

            # calculate total matrix by multiplying the single matrices
            matrix = np.dot(transl, np.dot(scale, rotation))

        else:
            raise ValueError("Invalid Value for n_dim. Got %d, but expected one "
                             "from [1, 2, 3]" % n_dim)

        return matrix

    def _build_matrix(self, n_dim=None):
        """
        Builds the actual matrix by trying to infer the dimensionality,
        resolve the default arguments and ensemble the matrix

        Parameters
        ----------
        n_dim : int
            the number of dimensions, try to extract it from
            given transformation parameters if not provided

        Returns
        -------
        :class:`np.ndarray`
            the transformation matrix

        """

        if n_dim is None:
            # check dimensionality if necessary
            if isinstance(self._scale, (tuple, list, np.ndarray)):
                n_dim = len(self._scale)
            elif isinstance(self._translation, (tuple, list, np.ndarray)):
                n_dim = len(self._translation)
            elif isinstance(self._rotation, (tuple, list, np.ndarray)):
                n_dim = len(self._rotation)
                # if less then 3 dimensions: Less rotation axes than dimensions
                # 1d: No rotation (0 + 1 = 1d)
                # 2d: 1 rotation axis (1 + 1 = 2d)
                if n_dim < 3:
                    n_dim = n_dim + 1
            else:
                raise RuntimeError("Cannot estimate dimensionality based on "
                                   "given arguments")

        # resolve default arguments
        scale, translation, rotation = self._set_default_args(
            self._scale, self._translation, self._rotation, n_dim)

        # actual matrix assembly
        matrix = self._ensemble_trafo_matrix(scale, translation, rotation,
                                             n_dim)

        matrix = inv(matrix)
        return matrix

    def _apply_sample_trafo(self, sample: np.ndarray):
        """
        Applies the given trafo to a single data sample
        If the matrix has not been calculated yet, it will be calculated before
        applying it.

        Parameters
        ----------
        sample : :class:`np.ndarray`
            the data sample

        Returns
        -------
        :class:`np.ndarray`
            the transformed sample

        """
        if self._n_dim is None:
            n_dim = len(sample.shape) - 1
        else:
            n_dim = self._n_dim
            assert len(sample.shape) == n_dim + 1

        if self._matrix is None:
            self._matrix = self._build_matrix(n_dim)
        sample = np.moveaxis(sample, 0, -1)
        transformed_sample = affine_transform(input=sample,
                                              matrix=self._matrix,
                                              **self._kwargs)

        transformed_sample = np.moveaxis(transformed_sample, -1, 0)
        return transformed_sample

    def __add__(self, other):
        """
        Overloads the addition operator, to interpret given numpy arrays as
        affine transformation matrices

        Parameters
        ----------
        other : :class:`np.ndarray` or
        :class:`delira.data_loading.transforms.AbstractTransform`
            the transformation to combine with; if numpy array: a new affine
            transformation will be created with the array as transformation
            matrix before combining it.

        Returns
        -------
        :class:`delira.data_loading.transforms.AbstractTransform`
            the combined transform

        """
        if isinstance(other, np.ndarray):
            other = Affine(matrix=other, source_keys=self._source_keys,
                           destination_keys=self._destination_keys)

        return super().__add__(other)

    def inverse(self):
        """
        Returns an inverse affine transform

        Returns
        -------

        """
        if self._matrix is None:
            self._build_matrix()

        return Affine(matrix=inv(self._matrix))

    def __invert__(self):
        return self.inverse()

    def __inv__(self):
        return self.inverse()


class Resize(Affine):
    """
    Resize Transformation

    Notes
    -----
    All arguments, which are passed on a per-dimension basis, must be given
    w.r.t the following order: ([z,] y,) x

    See Also
    --------
    :class:`Affine`
    """
    def __init__(self, target_size: tuple, **kwargs):
        """

        Parameters
        ----------
        target_size : tuple or int
            the target_size
            if tuple: must contain 1 value per dimension
            if int: value will be used for all dimensions except channels
        **kwargs :
            additional keyword arguments

        """
        super().__init__(output_shape=target_size, matrix=None, **kwargs)
        self._target_size = target_size
        self._last_size = None

    def _apply_sample_trafo(self, sample: np.ndarray):
        """
        Applies the transformation and (if necessary) recomputes the
        transformation matrix.

        Parameters
        ----------
        sample : :class:`np.ndarray`
            the sample to transform

        Returns
        -------
        :class:`np.ndarray`
            the transformed sample

        """
        # move channel axis to last
        _sample = np.moveaxis(sample, 0, -1)

        # use target_size for each dimension except channels if int
        if isinstance(self._target_size, int):
            target_size = ([self._target_size] * (len(_sample.shape) - 1)
                           + [_sample.shape[-1]])
        else:
            target_size = self._target_size
        curr_shape = np.array(_sample.shape[:len(target_size)])

        # check whether to recompute the transformation matrix
        if self._last_size is None or not np.array_equal(curr_shape,
                                                         self._last_size):

            # setting the matrix to None triggers recomputation
            self._matrix = None
            self._scale = np.array(target_size) / curr_shape
            self._last_size = target_size

        return super()._apply_sample_trafo(sample)


class Rotate(Affine):
    """
    Rotate Images

    Notes
    -----
    All arguments, which are passed on a per-dimension basis, must be given
    w.r.t the following order: ([z,] y,) x

    See Also
    --------
    :class:`Affine`

    """
    def __init__(self,  rotation, degrees=False, **kwargs):
        """

        Parameters
        ----------
        rotation : int or Iterable
            the rotation angles
        degrees : bool
            whether the :param:`rotation` values are passed as degrees or rads,
            If True: values will be converted to rads
        **kwargs : additional keyword arguments
        """
        if degrees:
            rotation = np.deg2rad(rotation)

        super().__init__(rotation=rotation, matrix=None, **kwargs)


class Zoom(Affine):
    """
    Zoom Images

    Notes
    -----
    All arguments, which are passed on a per-dimension basis, must be given
    w.r.t the following order: ([z,] y,) x

    See Also
    --------
    :class:`Affine`
    """
    def __init__(self, scale, **kwargs):
        """

        Parameters
        ----------
        scale : int or Iterable
            the scale values;
            if Iterable: must contain 1 value per dimension
            if Int: value will be used for all dimensions
        **kwargs :
            additional keyword arguments
        """
        super().__init__(matrix=None, scale=scale, **kwargs)


class Shift(Affine):
    """
    Affine Transformation which shifts the image by a given offset

    Notes
    -----
    All arguments, which are passed on a per-dimension basis, must be given
    w.r.t the following order: ([z,] y,) x

    See Also
    --------
    :class:`Affine`

    """
    def __init__(self, shift, **kwargs):
        """

        Parameters
        ----------
        shift : Iterable or int or float
            the shift values;
            If float values and all given values are smaller  than 1: the
                values will be used as relative offsets;
            Else: values will be used as absolute offsets
        **kwargs :
            additional keyword arguments

        """

        super().__init__(matrix=None, translation=shift, **kwargs)
        shift = np.array(shift)
        relative = shift.dtype == np.float and (shift <= 1.).all()
        self._relative_offset = relative
        self._shift = shift
        self._last_size = None

    def _apply_sample_trafo(self, sample: np.ndarray):
        """
        Applies the actual transformation

        Parameters
        ----------
        sample : :class:`np.ndarray`

        Returns
        -------
        :class:`np.ndarray`
            the transformed sample
            
        """
        # move channel axis to back
        _sample = np.moveaxis(sample, 0, -1)

        # check for scalar values (and apply it for all dimensions but
        # channels if scalar)
        if np.isscalar(self._shift):
            shift = np.array([self._shift] * (len(_sample.shape) - 1))
        else:
            shift = self._shift

        curr_shape = np.array(_sample.shape)
        # check whether matrix has to be recomputed
        if (self._relative_offset or self._last_size is None
                or not np.array_equal(curr_shape, self._last_size)):

            # setting matrix to None triggers recomputation
            self._matrix = None
            self._translation = curr_shape * shift

        return super()._apply_sample_trafo(sample)


class RotateAboutCentre(Rotate):
    """
    Rotates a single image around it's centre.
    This is done by shifting the centre to the origin, applying the rotation
    and shifting back
    """
    def __init__(self,  rotation, degrees=False, **kwargs):
        """

        Parameters
        ----------
        rotation : int or Iterable
            the rotation angles
        degrees : bool
            whether the :param:`rotation` values are passed as degrees or rads,
            If True: values will be converted to rads
        **kwargs : additional keyword arguments
        """
        super().__init__(rotation=rotation, degrees=degrees, **kwargs)

        self._shift_trafo = None
        self._inverse_shift_trafo = None
        self._last_size = None

    @property
    def shift_trafo(self):
        return self._shift_trafo, self._inverse_shift_trafo

    @shift_trafo.setter
    def shift_trafo(self, new_trafo: Affine):
        self._shift_trafo = new_trafo
        self._inverse_shift_trafo = new_trafo.inverse()

    def _apply_sample_trafo(self, sample: np.ndarray):
        """
        Applies the actual transformation

        Parameters
        ----------
        sample : :class:`np.ndarray`

        Returns
        -------
        :class:`np.ndarray`
            the transformed sample

        """
        # move channel axis to back
        _sample = np.moveaxis(sample, 0, -1)

        # calculate current shape
        curr_shape = np.array(_sample.shape[:-1])

        # if necessary recalculate the shifting transform
        if self._last_size is None or not np.array_equal(curr_shape,
                                                         self._last_size):
            self.shift_trafo = Shift(-curr_shape/2)
            self._last_size = curr_shape

        sample = self._shift_trafo._apply_sample_trafo(sample)
        sample = super()._apply_sample_trafo(sample)
        sample = self._inverse_shift_trafo._apply_sample_trafo(sample)

        return sample
