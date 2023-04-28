from epyr.circuit import Circuit
from epyr.operators import *
from epyr.state import * 

from numpy import array_equal


def test_simple_circuit():
    c = Circuit(1)  # 1-qubit circuit
    c.h()  # Add a Hadamard gate

    s = State(1)  # 1-qubit state |0>
    c.compute(s)
    assert s == plus

def test_Bell_State_creation():
    c = Circuit(2)  # 2-qubit circuit
    c.h(0)  # Add a Hadamard on the first qubit
    # Add a CNOT gate with the first qubit being the control and the second qubit being the target
    c.cnot(0, 1)

    s = State(2)  # |00>
    c.compute(s)
    assert s == phi_plus
