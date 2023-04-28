import numpy as np


__all__ = ["I", "X", "Y", "Z", "H", "S", "CNOT"]

# Define common quantum logic gates
# Pauli Gates
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# Clifford Gates
H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
S = np.array([[1, 0], [0, 1j]])
# Controlled NOT (CNOT) Gate (2-qubit operator)
CNOT = np.zeros((4, 4))
CNOT[0:2, 0:2] = I
CNOT[2:4, 2:4] = X

operator_dict = dict({
    "I": I,
    "X": X,
    "Y": Y,
    "Z": Z,
    "H": H,
    "S": S,
    "CNOT": CNOT,
})


class Operator:
    """A class that provides many common operators, as well as methods to create new operators."""

    def __init__(self):
        """Representation of an operator, or quantum logic gate."""
        raise NotImplemented

    @staticmethod
    def Itensor(n):
        """Returns the matrix (ndarray) corresponding to I^{(tensor) n}"""
        return np.eye(2 ** n)

    @staticmethod
    def create_controlled_gate(n, control, target, U):
        """Create a controlled U-gate on an n-qubit system."""
        raise NotImplemented
