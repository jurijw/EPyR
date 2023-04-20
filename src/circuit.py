import numpy as np
from state import State


class Circuit:
    """Represents a quantum circuit."""
    def __init__(self):
        raise NotImplemented

    def measure(state: State):
        """Perform a compledte measurement on the state. Returns the index of the basis state to which the wave-function collapses."""
        n = np.log2(state.arr.shape[0])
        indices = np.arange(2**n)
        probs = state.probabilities[:,0]
        collapsed_state_index = np.random.choice(indices, p=probs)
        return collapsed_state_index

    def create_circuit_unitary(gates):
        """Construct the quantum circuit unitary operator from an array of unitary gates."""
        U = gates[0]
        for i in range(1, gates.shape[0]):
            U = gates[i] @ U  # TODO: consider, if numpy has a built-in to do this.
        return U
