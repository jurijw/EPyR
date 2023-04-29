import numpy as np

from epyr.circuit import Circuit
from epyr.state import *

# Define 1 / sqrt(2) for convenience
INV2 = 1 / np.sqrt(2)


def configure(N):
    """Setup an N-qubit circuit and state."""
    c = Circuit(N)
    s = State(N)
    return c, s


def test_simple_circuit():
    c = Circuit(1)  # 1-qubit circuit
    c.h()  # Add a Hadamard gate

    s = State(1)  # 1-qubit state |0>
    c.compute(s)
    assert s == plus


def test_simple_multi_qubit_circuitS():
    expected = [
        np.array([1, 1]),
        np.array([1, 1, 0, 0]),
        np.array([1, 1, 0, 0, 0, 0, 0, 0])
    ]
    for N in range(1, 4):
        c = Circuit(N)  # N-qubit circuit
        c.h()  # Apply Hadamard on 1st qubit

        s = State(N)  # |..0>
        c.compute(s)
        assert s == INV2 * expected[N - 1]


def test_single_hadamard():
    expected = [
        np.array([1, 1, 0, 0, 0, 0, 0, 0]),
        np.array([1, 0, 1, 0, 0, 0, 0, 0]),
        np.array([1, 0, 0, 0, 1, 0, 0, 0])
    ]
    for i in range(3):
        c = Circuit(3)  # N-qubit circuit
        c.h(i)  # Apply Hadamard on ith qubit

        s = State(3)  # |000>
        c.compute(s)
        assert s == INV2 * expected[i]


def test_multiple_single_qubit_gates():
    expected = INV2 * np.array([0, 0, 0, 0, 0, 0, 1j, 1j])
    c, s = configure(3)
    c.h(0)
    c.x(1)
    c.y(2)
    c.compute(s)
    assert s == expected


def test_simple_multiple_single_qubit_gates():
    expected = INV2 * np.array([0, 0, 1, 1, 0, 0, 0, 0])
    c, s = configure(3)
    c.h(0)
    c.x(1)
    c.compute(s)
    assert s == expected


def test_bell_state_creation():
    c, s = configure(2)
    c.h(0)  # Add a Hadamard on the first qubit
    # Add a CNOT gate with the first qubit being the control and the second qubit being the target
    c.cnot(0, 1)
    c.compute(s)
    assert s == phi_plus


def test_cnot_skip():
    expected = INV2 * np.array([1, 0, 0, 0, 0, 1, 0, 0])
    c, s = configure(3)
    c.h(0)
    c.cnot(0, 2)
    c.compute(s)
    assert s == expected


def test_swapped_cnot():
    c, s, = configure(2)
    c.h(1)
    c.cnot(1, 0)
    c.compute(s)
    assert s == phi_plus


def test_hadamard_on_entangled_state():
    expected = 1 / 2 * np.array([1, 1, 1, -1])
    # Setup entangled state
    c = Circuit(2)
    s = State.common("phi_plus")  # 1 / sqrt(2) [|00> + |11>]
    c.h(0)
    c.compute(s)
    assert s == expected


def test_very_entangled_state():
    c, s = configure(4)
    c.h(0)
    c.cnot(0, 1)
    c.cnot(0, 2)
    c.cnot(0, 3)
    c.compute(s)

    expected = np.zeros(2 ** 4)
    expected[0], expected[2 ** 4 - 1] = 1, 1
    expected *= INV2  # 1 / sqrt(2) [|0000> + |1111>]
    assert s == expected


def test_very_very_entangled_state():
    c, s = configure(10)
    c.h(0)
    for i in range(1, 10):
        c.cnot(0, i)

    c.compute(s)

    expected = np.zeros(2 ** 10)
    expected[0], expected[2 ** 10 - 1] = 1, 1
    expected *= INV2  # 1 / sqrt(2) [|00...00> + |11...11>]
    assert s == expected


def test_very_very_entangled_states():
    for N in range(1, 20):
        c, s = configure(N)
        c.h(0)
        for i in range(1, N):
            c.cnot(0, i)

        c.compute(s)

        expected = np.zeros(2 ** N)
        expected[0], expected[2 ** N - 1] = 1, 1
        expected *= INV2  # 1 / sqrt(2) [|00...00> + |11...11>]
        assert s == expected

