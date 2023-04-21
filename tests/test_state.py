from epyr.state import State

def test_initialization():
    s = State(2)  # Initialize a 2 qubit state
    assert s.arr.shape == (2**2, 1)
    assert s.ket


