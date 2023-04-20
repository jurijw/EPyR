import numpy as np
from qvector import QVector


class State(QVector):
    def __init__(self, n: int) -> None:
        """Create an N qubit state, which is represented as a (ket) vector with 2^N entries. The vector is taken to be in the
        standard (computational) basis. By default, the state is initialized to the 0th basis state: |0...›.
        The state is represented by a (2^N, 1) np.ndarray."""

        state: np.ndarray = np.zeros(2**n, 1)
        state[0,:] = 1
        super.__init__(state)

    @staticmethod
    def basis_vector_string(n: int, i: int):
        """Returns a string corresponding to the ith basis vector in the n-qubit computational basis."""
        assert 0 <= i < 2**n, "i,n must satisfy 0 <= i < 2**n"
        assert type(n) == int, "n is not of type int"
        assert type(i) == int, "i is not of type int"
        i_base2: str = bin(i).split('b')[1]
        fill_zeros: str = (n - len(i_base2)) * '0'
        string = '|' + fill_zeros + i_base2 + '>'
        
        return string

    @staticmethod
    def ket_string(arr: np.ndarray):
        """Returns a representation of the passed array as a sum of basis states in braket notation."""
        nrows = len(arr)  # TODO: be careful of what happens when a bra is passed.
        n = int(np.log2(nrows))
        string = ""
        for i in range(nrows):
            ci = arr[i,0]  # The probability amplitude for the ith basis state.
            if ci == 0:
                continue
            else:
                string += f"({ci})"
            string += State.basis_vector_string(n, i)
            string += " "
        return string

    def show(self):
        """Print the state in braket notation in the computational basis."""
        print(State.ket_string(self.arr))

