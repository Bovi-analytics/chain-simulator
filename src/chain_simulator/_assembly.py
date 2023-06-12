"""Implementations of :mod:`~chain_simulator.abstract`."""
from typing import Iterator, Tuple, TypeVar

from scipy.sparse import coo_array

_T = TypeVar("_T", int, float)


def array_assembler(
        state_count: int, probability_calculator: Iterator[Tuple[int, int, _T]]
) -> coo_array:
    """Assemble transition matrix using a state change probability generator.

    Assemble a transition matrix in SciPy Coordinate (COO) array format using a
    state change probability generator. The assembler iterates over the
    generator, collects it's output and turns this into a SciPy COO array.

    Parameters
    ----------
    state_count : int
        How many states the probability generator yields.
    probability_calculator : Iterator[Tuple[int, int, _T]]
        Generator object yielding state change probabilities.

    Returns
    -------
    coo_array
        Transition matrix in SciPy COO array format.

    See Also
    --------
    chain_simulator.simulation.state_vector_processor
        Simulate a Markov chain and return intermediate/final state vector(s).

    Notes
    -----
    This assembler iterates oer the probability generator, collects its output
    and turns this into a SciPy COO array. The generator should yield the
    following items:

    - row index of probability (int).
    - column index of probability (int).
    - actual probability itself (int or float).

    Examples
    --------
    Example with a dummy generator which yields probabilities along a diagonal:
    >>> def dummy_generator():
    ...     for index in range(3):
    ...         yield index, index, index + 1
    >>> transition_matrix = array_assembler(3, dummy_generator())
    >>> transition_matrix
    <3x3 sparse array of type '<class 'numpy.int32'>'
        with 3 stored elements in COOrdinate format>
    >>> transition_matrix.toarray()
    array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]])
    """
    rows, cols, probabilities = [], [], []
    for row, col, probability in probability_calculator:
        rows.append(row)
        cols.append(col)
        probabilities.append(probability)
    return coo_array(
        (probabilities, (rows, cols)), shape=(state_count, state_count)
    )
