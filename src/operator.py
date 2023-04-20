import numpy as np

class Operator:
    """A class that provides many common operators, as well as methods to create new operators."""
    
    # Define common quantum logic gates (single qubit operators)
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = 1/np.sqrt(2) * np.array([[1, 1],[1, -1]])

    # Controlled NOT (CNOT) Gate (2-qubit operator)
    CNOT = np.zeros((4,4))
    CNOT[0:2,0:2] = I
    CNOT[2:4,2:4] = X
    
    def __init__(self):
        """Representation of an operator, or quantum logic gate."""
        raise NotImplemented

    @staticmethod
    def Itensor(n):
        """Returns the matrix (ndarray) corresponding to I^{\otimes n}"""
        return np.eye(2**n)

    @staticmethod
    def create_controlled_gate(n, control, target, U):
        """Create a controlled U-gate on an n-qubit system."""
        raise NotImplemented