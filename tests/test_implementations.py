"""Tests for module :mod:`~chain_simulator.implementations`."""
import sys
from typing import Iterator, Tuple

from chain_simulator._assembly import (
    array_assembler,
)
from chain_simulator._simulation import chain_simulator
from scipy.sparse import coo_array

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


def dummy_probability_generator() -> Iterator[Tuple[int, int, float]]:
    """Generate dummy probabilities.

    :return: Simple dummy probability generator.
    :rtype: Iterator[tuple[int, int, float]]
    """
    indices = range(4)
    values = [0.0, 0.0, 1.0, 0.5]
    for index, value in zip(indices, values):
        yield index, index, value


class TestArrayAssembler:
    """Tests for :func:`~chain_simulator.implementations.array_assembler`."""

    def test_assembly(self: Self) -> None:
        """Test simple dummy generator."""
        generator = dummy_probability_generator()
        result = array_assembler(4, generator)
        expected = coo_array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.5],
            ]
        )
        comparison = result != expected
        assert comparison.size <= 0


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
