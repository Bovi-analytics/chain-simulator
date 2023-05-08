"""Tests for module :mod:`~chain_simulator.implementations`."""
import sys
from typing import Iterator, Tuple

from chain_simulator._assembly import array_assembler
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
