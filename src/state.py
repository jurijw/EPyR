import numpy as np
from qvector import QVector


class State(QVector):
    def __init__(self, n: int) -> None:
        """Create an N qubit state, which is represented as a (ket) vector with 2^N entries. The vector is taken to be in the
        standard (computational) basis. By default, the state is initialized to the 0th basis state: |0...›.
        The state is represented by a (2^N, 1) np.ndarray."""

        state: np.ndarray = 0 * np.ones(2**n, 1)
        state[0] = 1
        super.__init__(state)

