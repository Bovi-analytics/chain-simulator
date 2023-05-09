from typing import Any, Iterator, Optional, Tuple, TypeVar

from numpy.typing import NDArray
from scipy.sparse import (
    csc_array,
    csc_matrix,
    csr_array,
    csr_matrix,
)

_T = TypeVar("_T", NDArray[Any], csc_array, csc_matrix, csr_array, csr_matrix)


def chain_simulator(
    transition_matrix: _T, steps: int, interval: Optional[int] = None
) -> Iterator[Tuple[_T, int]]:
    """Progress a Markov chain forward in time.

    Method which progresses a Markov chain forward in time using a provided
    transition matrix. Based on the `steps` parameter, the transition matrix is
    raised to the power of `steps`. This is done using a matrix multiplication.

    :param transition_matrix: Transition matrix.
    :type transition_matrix: coo_array
    :param steps: Steps in time to progress the simulation.
    :type steps: int
    :return: Transition matrix progressed in time.
    :rtype coo_array
    """
    # Validate `steps` and `interval` parameters for negative values.
    if steps <= 0:
        raise ValueError("Value of parameter `steps` must be higher than 0!")
    if interval and interval <= 0:
        raise ValueError(
            "Value of parameter `interval` must be higher than 0!"
        )

    progressed_matrix = transition_matrix.copy()
    step_range = range(1, steps + 1)
    for step in step_range:
        progressed_matrix = progressed_matrix.dot(transition_matrix)
        if interval and step < step_range[-1] and step % interval == 0:
            yield progressed_matrix, step
    yield progressed_matrix, step_range[-1]
