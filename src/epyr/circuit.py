import numpy as np
from state import State
from operators import operator_dict
from epyr_exception import EpyrException 


class Circuit:
    """Represents a quantum circuit."""

    def __init__(self, N):
        """Create an n-qubit quantum circuit."""
        self._n = N
        self._gates = []
        # Unitary transform corresponding to the entire circuit.
        # Note: this is incredibly inefficient.
        self._U = None  

    @property
    def N(self):
        """Return the number of qubits this circuit acts on."""
        return self._n

    @property
    def U(self):
        return self._U

    @property
    def gates(self):
        """Return a list of tuples, containing the gates this circuit is composed of,
        as well as the indices of the qubits they act upon. The gates are listed 
        sequentially, in the order they will be applied."""
        return self._gates

    def measure(self, state: State):
        """Perform a complete measurement on the state. Returns the index of the basis state to which the wave-function
        collapses."""
        basis_indices = np.arange(2 ** self.N)
        probabilities = state.probabilities[:, 0]
        collapsed_state_index = np.random.choice(basis_indices, p=probabilities)
        return collapsed_state_index

    @staticmethod
    def create_circuit_unitary(gates):
        """Construct the quantum circuit unitary operator from an array of unitary gates. Note: again, this is incredibly
        inefficient."""
        U = gates[0]
        for i in range(1, len(gates)):
            U = gates[i] @ U  # TODO: consider, if numpy has a built-in to do this.
        return U

    def add(self, gate, indices):
        """Add a gate, sequentially to my circuit's gates. The gate passed must be one of the common gates
        defined in the operators module, passed as a string, or an (n x n) unitary. Indices can be a single index or a list of
        indices, which the gate should be applied to."""
        # TODO: check unitarity.
        if type(gate) == str:
            if gate not in operator_dict.keys():
                raise EpyrException("The requested gate is not available.")
        # TODO: check unitarIty and that it functions with the indices
        self._gates.append((gate, indices))

    def compute(self, state):
        """Apply the quantum circuit to the given input state."""
        self._U = self.create_circuit_unitary(self.gates)
        return self.U @ state.arr
    
    def efficient_compute(self, state):
        """Apply the quantum circuit to the given input state."""
        raise NotImplemented
    
def apply_general_one_qubit_gate_in_place(state, U, target_index, N):
    """
    Applies a 1-qubit quantum gate U to a state vector. Mutates the state vector 
    in place, so as to avoid larger matrix multiplications. Algorithm is 
    linear in the number of entries in the state vector, in terms of both time 
    and space.

    Runtime complexity: O(2^N) 
    Space complexity:   O(2^N) 

    state:              a vector of length 2^N, where the ith entry gives the
                        probability amplitude to measure the system in the ith 
                        basis state. (In the computational basis)
    U:                  a 2x2 unitary matrix, representing the 1-qubit gate.
    target_index:       the index of the qubit on which the gate is applied.
                        Indexing from 0.
    N:                  the number of qubits
    """
    # TODO: consider checking if U is diagonal.
    pair_index_delta = 2 ** target_index
    jump_size = 2 ** (target_index + 1)
    num_jumps = 2 ** (N - (target_index + 1))

    u00, u01 = U[0]
    u10, u11 = U[1]
    # Iterate over the basis states. The gate acts on pairs of
    # basis states, whose binary representation differs exactly
    # in the bit with index target_index.
    for m in range(num_jumps):
        for n in range(pair_index_delta):
            j = m * jump_size + n
            j_prime = j + pair_index_delta
            alpha_j = state[j]
            alpha_j_prime = state[j_prime]
            state[j] = alpha_j * u00 + alpha_j_prime * u01
            state[j_prime] = alpha_j * u10 + alpha_j_prime * u11


    
def apply_general_two_qubit_gate_in_place(state, U, q0, q1, N):
    """
    Apply the two-qubit gate U to qubits with index q0 and q1 for
    an N qubit state. Performs this operation in-place, mutating
    the state vector, to avoid large matrix multiplication.
    """
    assert q0 < q1
    # TODO: if q1 < q0, then transpose U?
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