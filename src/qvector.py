import numpy as np
from typing import Union


class QVector:
    """A wrapper class for the beloved numpy array, representing quantum state vectors, which implements some convenience
    functions for quantum computing purposes."""

    def __init__(self, arr: Union[np.ndarray, list], ket: bool = True) -> None:
        """
        Creates a QVector instance, which resembles the passed ARR.
        Ensures the state is either in the shape (N, 1) representing a Ket,
        or in the shape (1, N) representing a Bra. Also normalizes the vector.
        """

        if type(arr) == list:
            # Convert the list to a numpy array
            arr = np.array([arr])


        try:
            rows, cols = arr.shape
            if not (rows == 1 or cols == 1):
                raise ValueError("Input array must be of the shape (1,N) or (N,1).")
        except ValueError:
            # Triggered when an ndarray of shape (N,) is passed
            arr = arr.reshape(arr.shape[0], 1)

        if ket:
            arr = arr.reshape(rows, 1)
        else:
            arr = arr.reshape(1, rows)
        # Normalize the state vector
        arr /= np.sum(arr)

        self._ket = ket
        self._arr = arr

    @property  # TODO: Perhaps return a copy of array. -> Probably not good from a performance stand-point.
    def arr(self):
        """Return the np.ndarray I represent."""
        return self._arr

    @property
    def shape(self):
        """Return the shape of the np.ndarray I represent."""
        return self._arr.shape

    @property
    def ket(self):
        """Returns True iff I represent a Ket"""
        return self._ket

    def dagger(self):
        """Perform the conjugate (Hermitian) transpose on my vector."""
        self._arr = self._arr.conj().T
        self._ket = not self._ket

    def print(self):
        """Prints the vector using Dirak (Bra-Ket) notation."""
        raise NotImplemented
