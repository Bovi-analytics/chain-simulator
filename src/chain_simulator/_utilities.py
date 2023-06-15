"""Useful utilities that can help perform common tasks."""
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Tuple, TypeVar
from warnings import warn

import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

if TYPE_CHECKING:
    from chain_simulator._simulation import STATE_VECTOR

_T = TypeVar(
    "_T",
    NDArray[Any],
    sps.coo_array,
    sps.coo_matrix,
    sps.csc_array,
    sps.csc_matrix,
    sps.csr_array,
    sps.csr_matrix,
)

_logger = logging.getLogger(__name__)


class TransitionMatrixWarning(Warning):
    pass


class TransitionMatrixSumWarning(TransitionMatrixWarning):
    pass


class TransitionMatrixNegativeWarning(TransitionMatrixWarning):
    pass


class NoCallbackWarning(Warning):
    pass


def validate_matrix_sum(transition_matrix: _T) -> bool:
    """Validate the sum of every row in a transition matrix.

    Checks whether every row in `transition_matrix` sums to 1. In this case
    the function evaluates TRUE. If one or more rows do not sum to 1, the
    function evaluates FALSE. Every row that does not sum to 1 is logged
    for troubleshoting.

    Parameters
    ----------
    transition_matrix : Any 2d NumPy array or SciPy COO/CSC/CSR array/matrix
        A Markov chain transition matrix.

    Returns
    -------
    bool
        Indication whether all rows in `transition_matrix` sum to 1.

    Warns
    -----
    TransitionMatrixSumWarning
        If there are any rows that do not sum t exactly 1.

    See Also
    --------
    validate_matrix_positive :
        Validate all probabilities in a transition matrix for a positive sign.

    Notes
    -----
    All rows in a transition matrix always sum to exactly 1. Method of
    validation: first the sum is computed for each row in the transition
    matrix. Then, the count of rows in the transition matrix is subtracted
    from the sum of sums of all rows. This subtraction should be exactly 0
    when the transition matrix is valid. When this subtraction is not exactly
    0, the transition matrix is faulty.

    Examples
    --------
    Validate a valid transition matrix:

    >>> from numpy import array
    >>> valid_transition_matrix = array(
    ...     [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]
    ... )
    >>> validate_matrix_sum(valid_transition_matrix)
    True

    Validate a faulty transition matrix where each row sums to 0:

    >>> faulty_transition_matrix = array(
    ...     [[-1, 1, 0], [1, 0, -1], [0, -1, 1]]
    ... )
    >>> validate_matrix_sum(faulty_transition_matrix)
    False
    """
    _logger.debug(
        "Validating sums of rows of transition matrix %s.",
        transition_matrix.shape,
    )
    sum_rows = transition_matrix.sum(1)
    if not (is_valid := not len(sum_rows) - sum_rows.sum()):
        for index, sum_row in enumerate(sum_rows):
            if sum_row != 1:
                warn(
                    "Row %d sums to %f instead of 1!" % (index, sum_row),
                    TransitionMatrixSumWarning,
                    stacklevel=1,
                )
    else:
        _logger.info("Transition matrix is valid (all rows sum to 1).")
    return is_valid


def validate_matrix_negative(transition_matrix: _T) -> bool:
    """Validate probabilities in a transition matrix for negative signs.

    Checks whether all probabilities in `transition_matrix` are positive. In
    case all probabilities are positive, this function evaluates TRUE. In case
    one or more probabilities are negative, this function evaluates FALSE.

    Parameters
    ----------
    transition_matrix : Any 2d NumPy array or SciPy COO/CSC/CSR array/matrix
        A Markov chain transition matrix.

    Returns
    -------
    bool
        Indication whether all probabilities in `transition_matrix` are
        positive.

    Warns
    -----
    TransitionMatrixNegativeWarning
        If there are any negative probabilities.

    See Also
    --------
    validate_matrix_sum : Validate the sum of every row in a transition matrix.

    Notes
    -----
    All probabilities in a transition matrix must be positive. Method of
    validation: all probabilities in a transition matrix are checked if they
    are less than zero. Probabilities less than zero are counted. When this
    count is exactly 0, the transition matrix is valid. Otherwise, it is
    considered faulty.

    Examples
    --------
    Validate a valid transition matrix:

    >>> from numpy import array
    >>> valid_transition_matrix = array(
    ...     [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]
    ... )
    >>> validate_matrix_negative(valid_transition_matrix)
    True

    Validate a faulty transition matrix a negative probability in each row:

    >>> invalid_transition_matrix = array(
    ...     [[-1, 1, 0], [1, 0, -1], [0, -1, 1]]
    ... )
    >>> validate_matrix_negative(invalid_transition_matrix)
    False
    """
    _logger.debug(
        "Validating probability signs of transition matrix %s.",
        transition_matrix.shape,
    )
    negative = transition_matrix < 0
    try:
        is_valid = not negative.count_nonzero()  # type: ignore[attr-defined]
    except AttributeError:
        is_valid = not np.any(negative)
    if not is_valid:
        indices = np.argwhere(negative == 1)
        for index in indices:
            warn(
                "Probability on index %s is negative!" % index,
                TransitionMatrixNegativeWarning,
                stacklevel=1,
            )
    else:
        _logger.info(
            "Transition matrix is valid (all probabilities are positive)."
        )
    return is_valid


class CallbackError(Exception):
    pass


class AccumulationError(Exception):
    pass


def simulation_accumulator(
    vector_processor: "Iterator[Tuple[STATE_VECTOR, int]]",
    **callbacks: "Callable[[STATE_VECTOR, int], None | int | float]",
) -> "Dict[str, None | int | float]":
    """Accumulate simulation data using callback functions.

    Accumulate data from intermediate/final state vectors using callback
    functions. Callback functions must accept a state vector along with the
    current step in time and can return either nothing, an int or float. These
    returned values are accumulated and summed per callback function and are
    returned after the simulation has finished.

    Parameters
    ----------
    vector_processor : Iterator[Tuple[numpy.ndarray, int]]
        Iterator yielding a state vector and the current step in time.
    **callbacks : Callable[[numpy.ndarray, int], None | int | float]
        Callable accepting a state vector and the current step in time and
        returning either nothing, an int or float.

    Returns
    -------
    Dict[str, None | int | float]
        Dictionary with the name of the callback function and corresponding
        accumulated result.

    Raises
    ------
    TypeError
        If the processor or callback functions have incorrect types.
    CallbackError
        If the callback function raises an exception.
    AccumulationError
        If the return value from callback function cannot be summed.

    Warns
    -----
    NoCallbackWarning
        If there are no callback specified.

    Warnings
    --------
    This function serves as a wrapper for
    :func:`chain_simulator.simulation.state_vector_processor`. The creation of
    this processor should be handled by the caller of this wrapper function. It
    is important that the processor was not iterated upon before calling this
    wrapper function. Otherwise, some state vectors cannot be processed using
    this accumulator.

    See Also
    --------
    chain_simulator.simulation.state_vector_processor
        Simulate a Markov chain and return intermediate/final state vector(s).

    Notes
    -----
    Callback functions are accepted as keyword-arguments, meaning that they can
    be provided as key-value pairs or unpacked from a dictionary.

    Callback functions are not required to return anything. If nothing is
    returned, no accumulator is created. Make sure to access accumulators using
    `.get(<callback_name>)` instead of `[<callback_name>]` to check whether an
    accumulator was created.

    Callback functions should have the following signature::

        def callback_function(
            state_vector: numpy.ndarray, step_in_time: int
        ) -> None | int | float:
            ...

    Intermediate/final state vectors are provided as-is from the simulation.

    Examples
    --------
    Provide callbacks as keyword arguments:

    >>> from numpy import array, sum
    >>> from chain_simulator.simulation import state_vector_processor
    >>> state_vector = array([1, 0, 0])
    >>> transition_matrix = array(
    ...     [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]
    ... )
    >>> processor = state_vector_processor(
    ...     state_vector, transition_matrix, 4, 1
    ... )
    >>> accumulated = simulation_accumulator(
    ...     processor,
    ...     time_cumulative=lambda x, y: sum(x),
    ...     vector_sum=lambda x, y: y
    ... )
    >>> accumulated
    {'time_cumulative': 4.0, 'vector_sum': 10}

    Or add callbacks to a dictionary and unpack them in the accumulator:

    >>> from numpy import array, sum
    >>> from chain_simulator.simulation import state_vector_processor
    >>> state_vector = array([1, 0, 0])
    >>> transition_matrix = array(
    ...     [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]
    ... )
    >>> processor = state_vector_processor(
    ...     state_vector, transition_matrix, 4, 1
    ... )
    >>> callback_functions = {
    ...     "time_cumulative": lambda x, y: sum(x),
    ...     "vector_sum": lambda x, y: y
    ... }
    >>> accumulated = simulation_accumulator(
    ...     processor, **callback_functions
    ... )
    >>> accumulated
    {'time_cumulative': 4.0, 'vector_sum': 10}

    Callbacks using constant extra parameters can be passed using
    `functools.partial`:

    >>> from numpy import array
    >>> from chain_simulator.simulation import state_vector_processor
    >>> from functools import partial
    >>> from typing import Callable
    >>> state_vector = array([1, 0, 0])
    >>> transition_matrix = array(
    ...     [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]
    ... )
    >>> processor = state_vector_processor(
    ...     state_vector, transition_matrix, 4, 1
    ... )
    >>> partial_callable: Callable = partial(lambda x, y, z: y * z, z=5)
    >>> accumulated = simulation_accumulator(
    ...     processor, callable_with_constant=partial_callable
    ... )
    >>> accumulated
    {'callable_with_constant': 50}
    """
    # Check input variables to prevent unwanted errors.
    _logger.info("Start accumulating simulation data.")
    _logger.debug("Validating input parameters.")
    if not isinstance(vector_processor, Iterator):
        raise TypeError("Invalid vector_processor, expected an iterable.")
    if callbacks:
        for callback_name, callback in callbacks.items():
            if not isinstance(callback, Callable):
                raise TypeError(
                    f"Callback function `{callback}` for `{callback_name}` is "
                    f"not callable."
                )
    else:
        warn(
            "Nothing to accumulate, no callbacks specified.",
            NoCallbackWarning,
            stacklevel=1,
        )
    _logger.debug("All input parameters seem OK.")

    accumulated_values = {}
    for vector, step_in_time in vector_processor:
        _logger.debug(
            "Accumulating data for step {step_in_time} in time.",
            extra={"step_in_time": step_in_time},
        )
        for callback_name, callback in callbacks.items():
            try:
                callback_value = callback(vector, step_in_time)
            except Exception as err:
                _logger.exception(
                    "Callback function `{callback}` raised an exception."
                )
                raise CallbackError(
                    f"Something went wrong while calling function "
                    f"`{callback}`."
                ) from err
            if callback_value is not None:
                if callback_name in accumulated_values:
                    try:
                        accumulated_values[callback_name] += callback_value
                    except TypeError as err:
                        _logger.exception("summation of variables went wrong.")
                        raise AccumulationError(
                            f"Could not accumulate value from `{callback}` "
                            f"for `{callback_name}`."
                        ) from err
                else:
                    _logger.debug(
                        "Adding new accumulator for `{callback_name}`.",
                        extra={"callback_name": callback_name},
                    )
                    accumulated_values[callback_name] = callback_value
    _logger.info(
        "Done accumulating simulation data, {current} out of {to} callbacks "
        "accumulated.",
        extra={"current": len(accumulated_values), "to": len(callbacks)},
    )
    return accumulated_values


def _plot_coo_matrix(m):
    "https://stackoverflow.com/questions/22961541/python-matplotlib-plot-sparse-matrix-pattern"
    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='white')
    ax.plot(m.col, m.row, 's', color='black', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax
