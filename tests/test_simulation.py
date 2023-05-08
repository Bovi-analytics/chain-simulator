from typing import Self

from chain_simulator import chain_simulator
from scipy.sparse import coo_array


class TestChainSimulator:
    """Tests for :func:`~chain_simulator.implementations.chain_simulator`."""

    def test_matmul_1(self: Self) -> None:
        """Test matrix multiplication once."""
        array = coo_array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
        result = chain_simulator(array, 1)
        expected = coo_array(
            [[0.00, 0.50, 0.50], [0.00, 0.25, 0.75], [0.00, 0.00, 1.00]]
        )
        comparison = result != expected
        assert comparison.size <= 0

    def test_matmul_2(self: Self) -> None:
        """Test matrix multiplication twice."""
        array = coo_array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
        result = chain_simulator(array, 2)
        expected = coo_array(
            [
                [0.000, 0.250, 0.750],
                [0.000, 0.125, 0.875],
                [0.000, 0.000, 1.000],
            ]
        )
        comparison = result != expected
        assert comparison.size <= 0
