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

    MATRIX_DOT_SUPPORT = TypeVar(
        "MATRIX_DOT_SUPPORT",
        NDArray[Any],
        sparse.csc_array,
        sparse.csc_matrix,
        sparse.csr_array,
        sparse.csr_matrix,
        _cupyx.scipy.sparse.csc_matrix,
        _cupyx.scipy.sparse.csr_matrix,
    )
    STATE_VECTOR = TypeVar(
        "STATE_VECTOR",
        NDArray[Any],
        _cupy.ndarray,
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
    TRANSITION_MATRIX = TypeVar(
        "TRANSITION_MATRIX",
        NDArray[Any],
        sparse.csc_array,
        sparse.csc_matrix,
        sparse.csr_array,
        sparse.csr_matrix,
        sparse.coo_array,
        sparse.coo_matrix,
        _cupyx.scipy.sparse.csc_matrix,
        _cupyx.scipy.sparse.csr_matrix,
    )

__all__ = ["state_vector_processor"]


def chain_simulator(
    transition_matrix: "MATRIX_DOT_SUPPORT",
    steps: "int",
    interval: "Optional[int]" = None,
) -> "Iterator[Tuple[MATRIX_DOT_SUPPORT, int]]":
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
    (array([[0, 1 / 8, 7 / 8], [0, 1 / 16, 15 / 16], [0, 0, 1]]), 3)

    To get all intermediary results, we can use the parameter `interval`:

    >>> simulator = chain_simulator(transition_matrix, 3, interval=1)
    >>> next(simulator)
    (array([[0, 1 / 2, 1 / 2], [0, 1 / 4, 3 / 4], [0, 0, 1]]), 1)
    >>> next(simulator)
    (array([[0, 1 / 4, 3 / 4], [0, 1 / 8, 7 / 8], [0, 0, 1]]), 2)
    >>> next(simulator)
    (array([[0, 1 / 8, 7 / 8], [0, 1 / 16, 15 / 16], [0, 0, 1]]), 3)
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
    state_vector: "NDArray[Any]",
    transition_matrix: "SCIPY_SPARSE_MATRIX",
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
    state_vector: "_cupy.ndarray",
    transition_matrix: "CUPY_MATRIX",
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


def to_cupy_array(state_vector: "STATE_VECTOR") -> "_cupy.ndarray":
    if isinstance(state_vector, np.ndarray):
        return _cupy.array(state_vector)
    if isinstance(state_vector, _cupy.ndarray):
        return state_vector
    # TODO: add meaningful error message
    raise TypeError


def to_cupy_matrix(transition_matrix: "MATRIX_DOT_SUPPORT") -> "CUPY_MATRIX":
    if isinstance(transition_matrix, np.ndarray):
        return _cupy.array(transition_matrix)
    if isinstance(transition_matrix, (sparse.csc_array, sparse.csc_matrix)):
        return _cupyx.scipy.sparse.csc_matrix(transition_matrix)
    if isinstance(transition_matrix, (sparse.csr_array, sparse.csr_matrix)):
        return _cupyx.scipy.sparse.csr_matrix(transition_matrix)
    if isinstance(
        transition_matrix,
        (
            _cupy.ndarray,
            _cupyx.scipy.sparse.csc_matrix,
            _cupyx.scipy.sparse.csr_matrix,
        ),
    ):
        return transition_matrix
    # TODO: add meaningful error message
    raise TypeError


def state_vector_processor(
    state_vector: "STATE_VECTOR",
    transition_matrix: "TRANSITION_MATRIX",
    steps: "int",
    interval: "Optional[int]" = None,
) -> "Iterator[Tuple[NDArray[Any], int]]":
    """Simulate a Markov chain and return (intermediary) state vector(s).

    Dynamically simulate a Markov chain on either a Central Processing Unit
    (CPU) or Graphics Processing Unit (GPU). The `state_vector` is multiplied
    with the `transition_matrix`, returning an intermediate/final state vector
    of the n-th or `step`-th step in time. By default, only a final state
    vector is returned. Intermediate state vectors are obtained by setting
    `interval`, which will represent every n-th intermediate state vector.

    Parameters
    ----------
    state_vector : any 1D array
        A 1D array with an initial state probability distribution, i.e. an
        initial state vector.
    transition_matrix : any 2D array
        A 2D array with state change probabilities, i.e. a transition matrix.
    steps : int
        How many `steps` in time `transition_matrix` must progress.
    interval : int, optional
        Which n-th or `interval`-th intermediate state vector must be returned,
        none by default.

    Yields
    ------
    tuple of array and int
        An intermediate/final state vector of the current step in time and
        the current step in time.

    Raises
    ------
    TypeError
        If the transition matrix type is incompatible with a GPU.

    See Also
    --------
    chain_simulator : Progress a Markov chain forward in time.

    Notes
    -----
    There are three distinct implementations. The first implementation is
    GPU-based, implemented with CuPy. CuPy is however an optional dependency.
    If the library is not installed, or the state vector / transition matrix
    format is compatible with a GPU, the function falls back to a SciPy
    implementation. If the transition matrix is a NumPy array, the function
    falls back to the final implementation, which is implemented using NumPy.

    Vector-matrix multiplication is used with the `dot`-method. CuPy, SciPy and
    NumPy all have their own optimised version of this method, so it is
    important to choose the right implementation. Choosing a wrong
    implementation results in errors (types are incompatible) and lower
    performance (a dot product may be faster on a GPU than a CPU).

    Examples
    --------
    Simulate a Markov chain for 3 days with a NumPy transition matrix:

    >>> import numpy as np
    >>> initial_state_vector = np.array([1, 0, 0])
    >>> transition_matrix = np.array(
    ...     [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]
    ... )
    >>> simulator = state_vector_processor(
    ...     initial_state_vector, transition_matrix, 3
    ... )
    >>> next(simulator)
    (array([[0, 1 / 8, 7 / 8], [0, 1 / 16, 15 / 16], [0, 0, 1]]), 3)

    Simulate a Markov chain for 2 days with a SciPy transition matrix and all
    intermediary results:

    >>> from scipy import sparse
    >>> csc_transition_matrix = sparse.csc_array(transition_matrix)
    >>> simulator = state_vector_processor(
    ...     initial_state_vector, csc_transition_matrix, 2, steps=1
    ... )
    >>> next(simulator)
    (array([[0, 1 / 2, 1 / 2], [0, 1 / 4, 3 / 4], [0, 0, 1]]), 1)
    >>> next(simulator)
    (np.array([[0, 1 / 4, 3 / 4], [0, 1 / 8, 7 / 8], [0, 0, 1]]), 2)
    """
    if isinstance(transition_matrix, (sparse.coo_array, sparse.coo_matrix)):
        transition_matrix = transition_matrix.tocsr()
    # Make use of a GPU using CuPy when the library is installed.
    if _cupy_installed:
        cupy_vector = to_cupy_array(state_vector)
        try:
            cupy_matrix = to_cupy_matrix(transition_matrix)
        except TypeError as err:
            # TODO: add meaningful error message
            raise TypeError from err
        simulator = vector_processor_cupy(
            cupy_vector, cupy_matrix, steps, interval
        )
        for progressed_matrix, current_step in simulator:
            yield progressed_matrix.get(), current_step
    elif sparse.issparse(transition_matrix):
        simulator = vector_processor_scipy(
            state_vector, transition_matrix, steps, interval
        )
    else:
        simulator = vector_processor_numpy(
            state_vector, transition_matrix, steps, interval
        )
    for result in simulator:
        yield result
