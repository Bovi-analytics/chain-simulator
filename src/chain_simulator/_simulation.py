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
    from typing import (
        Any,
        Iterator,
        Optional,
        Tuple,
        TypeVar,
    )

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
    SCIPY_SPARSE_MATRIX = TypeVar(
        "SCIPY_SPARSE_MATRIX",
        sparse.csc_array,
        sparse.csc_matrix,
        sparse.csr_array,
        sparse.csr_matrix,
    )
    CUPY_MATRIX = TypeVar(
        "CUPY_MATRIX",
        _cupy.ndarray,
        _cupyx.scipy.sparse.csc_matrix,
        _cupyx.scipy.sparse.csr_matrix,
    )


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


def vector_processor_numpy(
    state_vector: "NDArray[Any]",
    transition_matrix: "NDArray[Any]",
    steps: "int",
    interval: "Optional[int]" = None,
) -> "Iterator[Tuple[NDArray[Any], int]]":
    # Validate whether state vector and transition matrix are compatible types.
    is_numpy_array = isinstance(state_vector, np.ndarray)
    is_numpy_matrix = isinstance(transition_matrix, np.ndarray)
    if not all([is_numpy_array, is_numpy_matrix]):
        raise TypeError(
            f"State vector and transition matrix should be of types "
            f"`numpy.ndarray` and `numpy.ndarray` respectively, got "
            f"`{type(state_vector)}` and `{type(transition_matrix)}`!"
        )

    # Multiply state vector with transition matrix for new state vector.
    simulator = chain_simulator(transition_matrix, steps, interval)
    for progressed_matrix, current_step in simulator:
        yield np.dot(state_vector, progressed_matrix), current_step


def vector_processor_scipy(
    state_vector: "SCIPY_SPARSE_MATRIX",
    transition_matrix: "NDArray[Any]",
    steps: "int",
    interval: "Optional[int]" = None,
) -> "Iterator[Tuple[NDArray[Any], int]]":
    # Validate whether state vector and transition matrix are compatible types.
    is_numpy_array = isinstance(state_vector, np.ndarray)
    is_scipy_matrix = isinstance(
        transition_matrix,
        (
            sparse.csc_array,
            sparse.csc_matrix,
            sparse.csr_array,
            sparse.csr_matrix,
        ),
    )
    if not all([is_numpy_array, is_scipy_matrix]):
        raise TypeError(
            f"State vector and transition matrix should be of types "
            f"`numpy.ndarray` and any SciPy CSC/CSR array/matrix respectively,"
            f" got `{type(state_vector)}` and `{type(transition_matrix)}`!"
        )

    # Multiply state vector with transition matrix for new state vector.
    simulator = chain_simulator(transition_matrix, steps, interval)
    for progressed_matrix, current_step in simulator:
        yield sparse.spmatrix.dot(
            state_vector, progressed_matrix
        ), current_step


def vector_processor_cupy(
    state_vector: "CUPY_MATRIX",
    transition_matrix: "NDArray[Any]",
    steps: "int",
    interval: "Optional[int]" = None,
) -> "Iterator[Tuple[_cupy.ndarray, int]]":
    # Validate whether state vector and transition matrix are compatible types.
    is_cupy_array = isinstance(state_vector, _cupy.ndarray)
    is_cupy_matrix = isinstance(
        state_vector,
        (
            _cupy.ndarray,
            _cupyx.scipy.sparse.csc_matrix,
            _cupyx.scipy.sparse.csr_matrix,
        ),
    )
    if not all([is_cupy_array, is_cupy_matrix]):
        raise TypeError(
            f"State vector and transition matrix should be of types "
            f"`cupy.ndarray` and any CuPy ndarray/CSC/CSR matrix respectively,"
            f" got `{type(state_vector)}` and `{type(transition_matrix)}`!"
        )

    # Multiply state vector with transition matrix for new state vector.
    simulator = chain_simulator(transition_matrix, steps, interval)
    for progressed_matrix, current_step in simulator:
        yield _cupyx.scipy.sparse.spmatrix.dot(
            state_vector, progressed_matrix
        ), current_step


def state_vector_processor(state_vector, transition_matrix, steps, interval):
    if _cupy_installed:
        pass
