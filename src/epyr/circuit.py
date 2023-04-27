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
    
def apply_general_two_qubit_gate_in_place(state, U, q0, q1, N):
    """
    Apply the two-qubit gate U to qubits with index q0 and q1 for
    an N qubit state. Performs this operation in-place, mutating
    the state vector, to avoid large matrix multiplication.
    """
    for i0 in range(2 ** q1):
        for i1 in range(2 ** (q1 - q0 - 1)):
            for i2 in range(2 ** ((N - 1) - q1)):
                l = i0 + 2 ** (q0 + 1) * i1 + 2 ** (q1 + 1) * i2
                # Create a vector of relevant alpha_js
                # Below, j(b_q0)(b_q1) represents the index of the
                # basis state for fixed i0, i1, i2 and with the
                # bits in position q0 and q1 being b_q0 and b_q1 
                j00 = l + 2 ** q0 * 0 + 2 ** q1 * 0
                j01 = l + 2 ** q0 * 0 + 2 ** q1 * 1
                j10 = l + 2 ** q0 * 1 + 2 ** q1 * 0
                j11 = l + 2 ** q0 * 1 + 2 ** q1 * 1

                j = np.array([j00, j01, j10, j11])
                # Update all alpha_js by applying the U gate
                state[j] = U @ state[j]
                # Replace the alpha_js in the state vector