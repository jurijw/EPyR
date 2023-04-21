import numpy as np
from state import State
from operator_ import Operator


class Circuit:
    """Represents a quantum circuit."""

    def __init__(self, n):
        """Create an n-qubit quantum circuit."""
        self._n = n
        self._gates = []
        self._U = None  # Unitary transform corresponding to the entire circuit.

    @property
    def n(self):
        """Return the number of qubits this circuit acts on."""
        return self._n

    @property
    def U(self):
        return self._U

    @property
    def gates(self):
        """Return a list of the gates this circuit is composed of."""
        return self._gates

    def measure(self, state: State):
        """Perform a complete measurement on the state. Returns the index of the basis state to which the wave-function
        collapses."""
        basis_indices = np.arange(2 ** self.n)
        probabilities = state.probabilities[:, 0]
        collapsed_state_index = np.random.choice(basis_indices, p=probabilities)
        return collapsed_state_index

    @staticmethod
    def create_circuit_unitary(gates):
        """Construct the quantum circuit unitary operator from an array of unitary gates."""
        U = gates[0]
        for i in range(1, len(gates)):
            U = gates[i] @ U  # TODO: consider, if numpy has a built-in to do this.
        return U

    def add(self, gate):
        """Add a gate, sequentially to my circuit's gates. The gate passed must be an (n x n) unitary."""
        # TODO: check unitarity.
        self._gates.append(gate)

    def compute(self, state):
        """Apply the quantum circuit to the given input state."""
        self._U = self.create_circuit_unitary(self.gates)
        return self.U @ state.arr
