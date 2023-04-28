import numpy as np
from qvector import QVector

__all__ = ["up", "down", "plus", "minus", "right", "left", "phi_plus", "phi_minus", "psi_plus", "psi_minus", "State"]

# Common single qubit states
up = np.array([1, 0])
down = np.array([0, 1])
plus = 1/np.sqrt(2) * np.array([1, 1])
minus = 1/np.sqrt(2) * np.array([1, -1])
right = 1/np.sqrt(2) * np.array([1, 1j])
left = 1/np.sqrt(2) * np.array([1, -1j])

# Bell states
phi_plus = 1 / np.sqrt(2) * np.array([1, 0, 0, 1])
phi_minus = 1 / np.sqrt(2) * np.array([1, 0, 0, -1])
psi_plus = 1 / np.sqrt(2) * np.array([0, 1, 1, 0])
psi_minus = 1 / np.sqrt(2) * np.array([0, 1, -1, 0])

state_dict = dict({
    "up": up,
    "down": down,
    "plus": plus,
    "minus": minus,
    "right": right,
    "left": left,
    "phi_plus": phi_plus,
    "phi_minus": phi_minus,
    "psi_plus": psi_plus,
    "psi_minus": psi_minus,
})

class State:
    """A class that captures the state of a quantum system and provides utility functions."""

    def __init__(self, N: int) -> None:
        """Create an N qubit state, which is represented as a (ket) vector with 2^N entries. The vector is taken to be in the
        standard (computational) basis. By default, the state is initialized to the 0th basis state: |0...›.
        The state is represented by a (2^N,) np.ndarray."""

        self.state: np.ndarray = np.zeros(2 ** N)
        self.state[0] = 1
        self._N = N
    
    @property
    def N(self):
        return self._N

    @staticmethod
    def probabilities(arr):
        """Returns an array where the ith entry corresponds to the probability of measuring the state to be the ith basis state."""
        return np.real(arr.conj() * arr)

    def probabilities(self):
        """Returns an array where the ith entry corresponds to the probability of measuring my state to be the ith basis state."""
        State.probabilities(self.arr)

    @staticmethod
    def basis_vector_string(n: int, i: int):
        """Returns a string corresponding to the ith basis vector in the n-qubit computational basis."""
        assert 0 <= i < 2 ** n, "i,n must satisfy 0 <= i < 2**n"
        assert type(n) == int, "n is not of type int"
        assert type(i) == int, "i is not of type int"
        i_base2: str = bin(i).split('b')[1]
        fill_zeros: str = (n - len(i_base2)) * '0'
        string = '|' + fill_zeros + i_base2 + '>'

        return string

    @staticmethod
    def ket_string(state: np.ndarray):
        """Returns a representation of the passed array as a sum of basis states in braket notation."""
        N = int(np.log2(len(state)))
        string = ""
        for i in range(len(state)):
            # The probability amplitude for the ith basis state.
            ci = state[i] 
            if ci != 0:
                string += f"({ci})"
                string += State.basis_vector_string(N, i)
                string += " "
        return string

    def show(self):
        """Print the state in braket notation in the computational basis."""
        print(State.ket_string(self.state))
