"""Useful utilities that can help perform common tasks.

Module with small helper function for i.e. validating transition
matrices. These functions are written to help perform common tasks when
working with this package.
"""
import logging
from typing import Any, TypeVar
from warnings import warn

import numpy as np
import scipy.sparse as sps
from numpy.typing import NDArray

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
_logger.addHandler(logging.NullHandler())


class TransitionMatrixWarning(Warning):
    pass


class TransitionMatrixSumWarning(TransitionMatrixWarning):
    pass


class TransitionMatrixNegativeWarning(TransitionMatrixWarning):
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

    >>> import numpy as np
    >>> valid_transition_matrix = np.array(
    ...     [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]
    ... )
    >>> validate_matrix_sum(valid_transition_matrix)
    True

    Validate a faulty transition matrix where each row sums to 0:

    >>> faulty_transition_matrix = np.array(
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

    >>> import numpy as np
    >>> valid_transition_matrix = np.array(
    ...     [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]
    ... )
    >>> validate_matrix_negative(valid_transition_matrix)
    True

    Validate a faulty transition matrix a negative probability in each row:
    >>> invalid_transition_matrix = np.array(
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
            )
    else:
        _logger.info(
            "Transition matrix is valid (all probabilities are positive)."
        )
    return is_valid


def validate_matrix_positive(transition_matrix: _T) -> bool:
    warn(
        "Validator has been renamed to `validate_matrix_negative`, which "
        "provides the exact same functionality. This validator will be "
        "removed in a future version.",
        DeprecationWarning,
    )
    return validate_matrix_negative(transition_matrix)


validate_matrix_positive.__doc__ = validate_matrix_negative.__doc__
