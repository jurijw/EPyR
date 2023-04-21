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