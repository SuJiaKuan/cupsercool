from six import string_types

import numpy as np

class Blob(object):
    """The basic opeartion unit for streaming.

    Blob is the basic operation unit for streaming. A blob can contain zero, one
    or more float-like tensors (numpy `ndarrays` which are float-liketypes).
    """

    def __init__(self):
        """Create a new `Blob`."""
        self._data = dict()

    def has(self, name):
        """Check whether a tensor is in the blob or not.

        Args:
          name (string): The name of the tensor.

        Returns:
          bool: True if the tensor exists, False otherwise.

        Raises:
          TypeError: if `name` is not a string.
        """
        if not isinstance(name, string_types):
            raise TypeError('Tensor name must be a string.')
        return name in self._data

    def fetch(self, name):
        """Fetch a tensor from the blob.

        Args:
          name (string): The name of the tensor.

        Returns:
          Fetched tensor (numpy `ndarray`) if successful.

        Raises:
          TypeError: if `name` is not a string.
          RuntimeError: if the tensor does not exist in the blob.
        """
        if not isinstance(name, string_types):
            raise TypeError('Tensor name must be a string.')
        if not self.has(name):
            raise RuntimeError("Can't find tensor: {}".format(name))
        return self._data[name]

    def feed(self, name, tensor):
        """Feed a tensor into the blob.

        Args:
          name (string): The name of the tensor.
          tensor (numpy `ndarray`): A numpy `ndarray` object to fed into the
            blob. The types must be float-like (bool, int, unsigned
            int and float).

        Returns:
          Fed tensor (numpy `ndarray`) if successful.

        Raises:
          TypeError: if `name` is not a string or `tensor` is not a float-like
            tensor.
        """
        if not isinstance(name, string_types):
            raise TypeError('Tensor name must be a string.')
        if not isinstance(tensor, np.ndarray):
            raise TypeError('Only numpy ndarray is supported for feeding.')
        if tensor.dtype.kind not in 'biuf':
            raise TypeError('Only float-like types (bool, int, unsigned int and float) are supported for tensor.')

        self._data[name] = tensor

        return tensor
