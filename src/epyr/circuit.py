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
        probabilities = state.probabilities()
        collapsed_state_index = np.random.choice(
            basis_indices, p=probabilities)
        return collapsed_state_index

    @staticmethod
    def create_circuit_unitary(gates):
        """Construct the quantum circuit unitary operator from an array of unitary gates. Note: again, this is incredibly
        inefficient."""
        U = gates[0]
        for i in range(1, len(gates)):
            # TODO: consider, if numpy has a built-in to do this.
            U = gates[i] @ U
        return U

    ##################################
    ###### CIRCUIT CONSTRUCTION ######
    ##################################

    def add(self, gate, indices=None):
        """Add a gate, sequentially to my circuit's gates. The gate passed must be one of the common gates
        defined in the operators module, passed as a string, or an (m x m) unitary. Indices can be a single index or a list of
        indices, which the gate should be applied to."""
        # TODO: check unitarity.
        if type(gate) == str:
            if gate not in operator_dict.keys():
                raise EpyrException("The requested gate is not available.")
            gate = operator_dict[gate]

        # By default, apply the gate to the first log2(m) qubits.
        if indices is None:
            num_affected_qubits = int(np.log2(len(gate)))
            indices = [i for i in range(num_affected_qubits)]
        elif type(indices) == int:
            indices = [indices]
        # TODO: check unitarIty and that it functions with the indices
        self._gates.append((gate, indices))

    def add_common(self, gate, indices=None):
        """Add one of the common gates, defined in the operators module,
        to the circuit."""
        self.add(gate, indices)

    def i(self, index=0):
        """Add an identity gate to the qubit at position INDEX."""
        self.add_common("I", index)

    def x(self, index=0):
        """Add a NOT gate to the qubit at position INDEX."""
        self.add_common("X", index)

    def y(self, index=0):
        """Add a Y gate to the qubit at position INDEX."""
        self.add_common("Y", index)

    def z(self, index=0):
        """Add a Z gate to the qubit at position INDEX."""
        self.add_common("Z", index)

    def h(self, index=0):
        """Add a Hadamard gate to the qubit at position INDEX."""
        self.add_common("H", index)

    def s(self, index=0):
        """Add an S gate to the qubit at position INDEX."""
        self.add_common("S", index)

    def cnot(self, control, target):
        """Add a CNOT gate from the qubit at position CONTROL
        to the qubit at position TARGET."""
        indices = [control, target]
        self.add_common("CNOT", indices)

    def reset(self):
        """Clear all the gates in this circuit."""
        self._gates = []

    #############################
    ###### STATE EVOLUTION ######
    #############################

    def compute(self, state: State):  # TODO: Consider renaming state.state to state.vec(tor)
        """Apply the quantum circuit to the given input state."""
        for gate, indices in self._gates:
            # Check how many qubits are affected by the gate
            num_affected_qubits = int(np.log2(len(gate)))  # TODO: could be slow, consider computing in add() method.
            if num_affected_qubits == 1:
                target = indices[0]
                apply_general_one_qubit_gate_in_place(
                    state.state, gate, target, self.N)
            elif num_affected_qubits == 2:
                target0, target1 = indices
                apply_general_two_qubit_gate_in_place(
                    state.state, gate, target0, target1, self.N)
            else:
                raise NotImplemented


def apply_general_one_qubit_gate_in_place(state, U, q0, N):
    """
    Applies a 1-qubit quantum gate U to a state vector. Mutates the state vector
    in place to avoid larger matrix multiplications. Algorithm is
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
    for i0 in range(2 ** q0):
        for i1 in range(2 ** ((N - 1) - q0)):
                l = i0 + 2 ** (q0 + 1) * i1
                # Create a vector of relevant alpha_js
                # Below, j(b_q0)(b_q1) represents the index of the
                # basis state for fixed i0, i1, i2 and with the
                # bits in position q0 and q1 being b_q0 and b_q1
                j0 = l + 2 ** q0 * 0  # If q0 = 0
                j1 = l + 2 ** q0 * 1

                j = np.array([j0, j1])
                # Update all alpha_js by applying the U gate
                state[j] = U @ state[j]


def apply_general_two_qubit_gate_in_place(state, U, q0, q1, N):
    """
    Apply the two-qubit gate U to qubits with index q0 and q1 for
    an N qubit state. Performs this operation in-place, mutating
    the state vector, to avoid large matrix multiplication.
    """
    assert q0 < q1
    # TODO: if q1 < q0, then transpose U?
    for i0 in range(2 ** q0):
        # TODO: consider incrementing by the correct amount here rather than doing the multiplication below to get l.
        for i1 in range(2 ** (q1 - q0 - 1)):
            for i2 in range(2 ** ((N - 1) - q1)):
                l = i0 + 2 ** (q0 + 1) * i1 + 2 ** (q1 + 1) * i2
                # Create a vector of relevant alpha_js
                # Below, j(b_q0)(b_q1) represents the index of the
                # basis state for fixed i0, i1, i2 and with the
                # bits in position q0 and q1 being b_q0 and b_q1
                j00 = l + 2 ** q1 * 0 + 2 ** q0 * 0
                j01 = l + 2 ** q1 * 0 + 2 ** q0 * 1
                j10 = l + 2 ** q1 * 1 + 2 ** q0 * 0
                j11 = l + 2 ** q1 * 1 + 2 ** q0 * 1

                j = np.array([j00, j01, j10, j11])
                # Update all alpha_js by applying the U gate
                state[j] = U @ state[j]
                # Replace the alpha_js in the state vector
