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


def chain_simulator(array: coo_array, steps: int) -> coo_array:
    """Progress a Markov chain forward in time.

    Method which progresses a Markov chain forward in time using a provided
    transition matrix. Based on the `steps` parameter, the transition matrix is
    raised to the power of `steps`. This is done using a matrix multiplication.

    :param array: Transition matrix.
    :type array: coo_array
    :param steps: Steps in time to progress the simulation.
    :type steps: int
    :return: Transition matrix progressed in time.
    :rtype coo_array
    """
    compressed_array = array.tocsr()
    if steps == 1:
        return compressed_array @ compressed_array
    progressed_array = compressed_array
    for _step in range(steps):
        progressed_array = compressed_array @ progressed_array
    return progressed_array.tocoo()
