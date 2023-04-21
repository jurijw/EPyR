from epyr.circuit import Circuit
from epyr.operator_ import *
from epyr.state import State, plus

from numpy import array_equal


def test_simple_circuit():
    c = Circuit(1)  # 1-qubit circuit
    c.add(H)  # Add a Hadamard gate

    s = State(1)
    res = c.compute(s)
    assert array_equal(res, plus)


def test_Bell_State_creation():
    c = Circuit(2)  # 2-qubit circuit
    c.add(H, 0)  # Add a Hadamard on the first qubit
    # Add a CNOT gate with the first qubit being the control and the second qubit being the target
    c.cnot(0, 1)

    s = State(2)  # |00>
    res = c.compute(s)
    assert array_equal(res, State.phi_plus)
