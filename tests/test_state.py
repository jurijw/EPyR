import pytest
from state import State


def test_initialization():
    s = State(2)  # Initialize a 2 qubit state
    assert s.shape == (2, 1)


