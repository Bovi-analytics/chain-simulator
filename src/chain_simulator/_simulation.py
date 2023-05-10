from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

try:
    import cupy as _cupy
    import cupyx as _cupyx
except ImportError:
    _cupy_installed = False
    _cupy = None
    _cupyx = None
else:
    _cupy_installed = True

if TYPE_CHECKING:
    from typing import Any, Iterator, Optional, Tuple, TypeVar, Union

    from numpy.typing import NDArray

    _T = TypeVar(
        "_T",
        NDArray[Any],
        sparse.csc_array,
        sparse.csc_matrix,
        sparse.csr_array,
        sparse.csr_matrix,
        _cupyx.scipy.sparse.csc_matrix,
        _cupyx.scipy.sparse.csr_matrix,
    )
    _MATRIX_TYPES = Union[
        NDArray[Any],
        sparse.coo_array,
        sparse.coo_matrix,
        sparse.csc_array,
        sparse.csc_matrix,
        sparse.csr_array,
        sparse.csr_matrix,
    ]


def chain_simulator(
    transition_matrix: "_T", steps: "int", interval: "Optional[int]" = None
) -> "Iterator[Tuple[_T, int]]":
    """Progress a Markov chain forward in time.

    Progress a Markov chain by `steps` units of time. Each yield will contain
    two values: the progressed transition matrix and the current step in time.
    The final progressed transition matrix will always be returned, regardless
    of `interval`. If `interval` is specified, intermediate transition matrices
    will also be returned.

    Parameters
    ----------
    transition_matrix : NumPy 2d-array or SciPy csc/csr matrix/array
        A Markov chain transition matrix.
    steps : int
        How many steps in time the transition matrix must progress in time.
    interval : int, optional
        How often intermediary transition matrices are returned, none
        by default.

    Yields
    ------
    tuple of array and int
        A Markov chain transition matrix progressed in time and the current
        step in time.

    Raises
    ------
    ValueError
        If the value of parameter `steps` is less than 1.
    ValueError
        If the value of parameter `interval` is less than 1.

    Notes
    -----
    The function is implemented as a generator because of the need of
    intermediary transition matrices. Also, each yield contains the current
    step in time to indicate how far the transition matrix is progressed.
    Otherwise, callers need to do this bookbarking themselves.

    Examples
    --------
    To progress a transition matrix by 3 units of time, do the following:

    >>> import numpy as np
    >>> transition_matrix = np.array(
    ...     [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]
    ... )
    >>> simulator = chain_simulator(transition_matrix, 3)
    >>> next(simulator)
    (np.array([[0, 1 / 8, 7 / 8], [0, 1 / 16, 15 / 16], [0, 0, 1]]), 2)

    To get all intermediary results, we can use the parameter `interval`:

    >>> simulator = chain_simulator(transition_matrix, 3, interval=1)
    >>> next(simulator)
    (np.array([[0, 1 / 2, 1 / 2], [0, 1 / 4, 3 / 4], [0, 0, 1]]), 1)
    >>> next(simulator)
    (np.array([[0, 1 / 4, 3 / 4], [0, 1 / 8, 7 / 8], [0, 0, 1]]), 2)
    >>> next(simulator)
    (np.array([[0, 1 / 8, 7 / 8], [0, 1 / 16, 15 / 16], [0, 0, 1]]), 3)
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


def process_vector_numpy(
    transition_matrix, state_vector, steps, interval
):
    simulator = chain_simulator(transition_matrix, steps, interval)
    for progressed_matrix, current_step in simulator:
        yield np.dot(state_vector, progressed_matrix), current_step


def process_vector_scipy(
    transition_matrix, state_vector, steps, interval
):
    if transition_matrix.getformat() == "coo":
        transition_matrix = transition_matrix.tocoo()
    simulator = chain_simulator(transition_matrix, steps, interval)
    for progressed_matrix, current_step in simulator:
        yield sparse.spmatrix.dot(
            state_vector, progressed_matrix
        ), current_step


def process_vector_cupy(
    transition_matrix, state_vector, steps, interval
):
    if isinstance(transition_matrix, sparse.spmatrix):
        if (matrix_format := transition_matrix.getformat()) == "csc":
            cupy_matrix = _cupyx.scipy.sparse.csc_matrix(transition_matrix)
        elif matrix_format == "csr":
            cupy_matrix = _cupyx.scipy.sparse.csr_matrix(transition_matrix)
        elif matrix_format == "coo":
            cupy_matrix = _cupyx.scipy.sparse.csr_matrix(transition_matrix)
        else:
            raise TypeError
    elif isinstance(transition_matrix, np.ndarray):
        cupy_matrix = _cupy.array(transition_matrix)
    else:
        raise TypeError
    cupy_array = _cupy.array(state_vector)
    simulator = chain_simulator(cupy_matrix, steps, interval)
    for progressed_matrix, current_step in simulator:
        yield _cupyx.scipy.sparse.spmatrix.dot(
            cupy_array, progressed_matrix
        ).get(), current_step


def ArrayProcessor(
    transition_matrix: "_MATRIX_TYPES",
    state_vector: "NDArray[Any]",
    steps: "int",
    interval: "Optional[int]" = None,
) -> "Iterator[Tuple[NDArray[Any], int]]":
    if _cupy_installed:
        generator = process_vector_cupy(transition_matrix, state_vector, steps, interval)
    elif isinstance(transition_matrix, sparse.spmatrix):
        generator = process_vector_scipy(transition_matrix, state_vector, steps, interval)
    elif isinstance(transition_matrix, np.ndarray):
        generator = process_vector_numpy(transition_matrix, state_vector, steps, interval)
    else:
        raise TypeError
    for result in generator:
        yield result
