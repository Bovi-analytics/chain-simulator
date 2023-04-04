"""Useful utilities that can help perform common tasks.

Module with small helper function for i.e. validating transition
matrices. These functions are written to help perform common tasks when
working with this package.
"""
import logging
from typing import TypeVar

import numpy as np
from scipy.sparse import csr_array

_T = TypeVar("_T", np.array, csr_array)

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


def validate_matrix_sum(transition_matrix: _T) -> bool:
    """Validate the sum of every row in a transition matrix.

    All rows in a transition matrix always sum to exactly 1. The transition
    matrix in considered faulty.

    Method of validation: first the sum is computed for each row in the
    transition matrix. Then, the count of rows is divided by the sum of sums
    of all rows. This division should be exactly 1 when the transition matrix
    is valid. When the division is not exactly 1 the transition matrix is
    faulty.

    :param transition_matrix: any SciPy 2d-array or matrix.
    :type transition_matrix: _T
    :return: whether the sum of all rows in the transition matrix equal to 1.
    :rtype: bool
    """
    sum_rows = transition_matrix.sum(1)
    return (len(sum_rows) / sum_rows.sum()) == 1  # type: ignore[no-any-return]


def validate_matrix_positive(transition_matrix: _T) -> bool:
    """Validate if all numbers are positive in a transition matrix.

    Markov chains do not allow negative probabilities in transition matrices,
    although negative probabilities might slip in when calculating them. This
    function checks a transition matrix for negative probabilities.

    Method of validation: all probabilities in a transition matrix are checked
    if they are less than zero. Probabilities less than zero are counted. When
    this count is exactly 0, the transition matrix is valid. Otherwise, it is
    considered faulty.

    :param transition_matrix: any SciPy 2d-array or matrix.
    :type transition_matrix: _T
    :return: whether all numbers in the transition matrix are positive.
    :rtype: bool
    """
    values = transition_matrix < 0
    is_valid = values.size == 0
    if not is_valid:
        if _logger.isEnabledFor(logging.WARNING):
            indices = np.argwhere(values == 1)
            for index, value in zip(indices, transition_matrix[values]):
                _logger.warning(
                    "Probability %d with index %s is negative!", value, index
                )
    else:
        _logger.info(
            "Transition matrix is valid (all probabilities are positive)."
        )
    return is_valid
