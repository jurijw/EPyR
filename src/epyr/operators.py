import numpy as np

__all__ = ["I", "X", "Y", "Z", "H", "S", "CNOT", "SWAP"]


# Define inverse sqrt(2) for convenience
INV2 = 1 / np.sqrt(2)

# Define common quantum logic gates
# Pauli Gates
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# Clifford Gates
H = INV2 * np.array([[1, 1], [1, -1]])
S = np.array([[1, 0], [0, 1j]])

# T Gate
T = np.array([
    [1, 0],
    [0, np.exp(1j * np.pi / 4)]
])

# Controlled NOT (CNOT) Gate (2-qubit operator)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0]
])

SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

operator_dict = dict({
    "I": I,
    "X": X,
    "Y": Y,
    "Z": Z,
    "H": H,
    "S": S,
    "CNOT": CNOT,
    "SWAP": SWAP,
})


def swap_two_qubit_gate(gate):
    """Given a unitary 2x2 operator GATE, which operates on |q1 q0>,
    returns the operator equivalent of acting it on |q0 q1>. This is
    done by first applying a swap gate, then the operator, and then
    unswapping (SWAP^dagger = SWAP). TODO: there may be a more efficient way to do this.
    """
    return SWAP @ gate @ SWAP


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
