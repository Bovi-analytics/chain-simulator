"""Implementations of :mod:`~chain_simulator.abstract`."""
from decimal import Decimal
from typing import Iterator, Tuple, TypeVar

from scipy.sparse import coo_array

_T = TypeVar("_T", float, Decimal)


def array_assembler(
    state_count: int, probability_calculator: Iterator[Tuple[int, int, _T]]
) -> coo_array:
    """Assemble an array using a state change probability generator.

    Function which assembles a Coordinate (COO) array using a state change
    probability generator. Per probability this generator must provide the
    row index, column index and the probability itself in order to assemble
    an array.

    :param state_count: Number/count of all possible states.
    :type state_count: int
    :param probability_calculator: Generator to calculate probabilities.
    :type probability_calculator: Iterator[tuple[int, int, _T]]
    :return: Assembled array in Coordinate (COO) format.
    :rtype: coo_array
    """
    rows, cols, probabilities = [], [], []
    for row, col, probability in probability_calculator:
        rows.append(row)
        cols.append(col)
        probabilities.append(probability)
    return coo_array(
        (probabilities, (rows, cols)), shape=(state_count, state_count)
    )
