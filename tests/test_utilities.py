"""Tests for module :mod:`~chain_simulator.utilities`."""

import numpy as np
import numpy.typing as npt
import pytest
from _pytest.logging import LogCaptureFixture
from chain_simulator._utilities import (
    TransitionMatrixNegativeWarning,
    TransitionMatrixSumWarning,
    validate_matrix_negative,
    validate_matrix_sum,
)
from scipy.sparse import coo_array, csc_array, csr_array

TestingArray = npt.NDArray[np.int32]


def numpy_ndarray_valid() -> TestingArray:
    """Generate a valid NumPy ndarray.

    :return: A valid NumPy ndarray.
    :rtype: TestingArray
    """
    return np.array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])


def numpy_ndarray_zero() -> TestingArray:
    """Generate an invalid NumPy ndarray with only zeroes.

    :return: An invalid NumPy ndarray with zeroes.
    :rtype: TestingArray
    """
    return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def numpy_ndarray_negative() -> TestingArray:
    """Generate an invalid NumPy ndarray with negative numbers.

    :return: an invalid NumPy ndarray with negative numbers.
    :rtype: TestingArray
    """
    return np.array([[-1, 1, 0], [1, 0, -1], [0, -1, 1]])


def scipy_coo_array_valid() -> coo_array:
    """Generate a valid SciPy coo_array.

    :return: A valid SciPy coo_array.
    :rtype coo_array
    """
    return coo_array(numpy_ndarray_valid())


def scipy_coo_array_zero() -> coo_array:
    """Generate an invalid SciPy coo_array with only zeroes.

    :return: An invalid SciPy coo_array with zeroes.
    :rtype coo_array
    """
    return coo_array(numpy_ndarray_zero())


def scipy_coo_array_negative() -> coo_array:
    """Generate an invalid SciPy coo_array with negative numbers.

    :return: An invalid SciPy coo_array with negative numbers.
    :rtype: coo_array
    """
    return coo_array(numpy_ndarray_negative())


def scipy_csc_array_valid() -> csc_array:
    """Generate a valid SciPy csc_array.

    :return: A valid SciPy csc_array.
    :rtype: csc_array
    """
    return csc_array(numpy_ndarray_valid())


def scipy_csc_array_zeroes() -> csc_array:
    """Generate an invalid SciPy csc_array with only zeroes.

    :return: An invalid SciPy csc_array with zeroes.
    :rtype: csc_array
    """
    return csc_array(numpy_ndarray_zero())


def scipy_csc_array_negative() -> csc_array:
    """Generate an invalid SciPy csc_array with negative numbers.

    :return: An invalid SciPy csc_array with negative numbers.
    :rtype: csc_array
    """
    return csc_array(numpy_ndarray_negative())


def scipy_csr_array_valid() -> csr_array:
    """Generate a valid SciPy csr_array.

    :return Valid SciPy csr_array.
    :rtype: csr_array
    """
    return csr_array(numpy_ndarray_valid())


def scipy_csr_array_zero() -> csr_array:
    """Generate an invalid SciPy csr_array with only zeroes.

    :return: Invalid SciPy csr_array with only zeroes
    :rtype: csr_array
    """
    return csr_array(numpy_ndarray_zero())


def scipy_csr_array_negative() -> csr_array:
    """Generate an invalid SciPy csr_array with negative numbers.

    :return: Invalid SciPy csr_array with negative numbers.
    :rtype: csr_array
    """
    return csr_array(numpy_ndarray_negative())


class TestTransitionMatrixSum:
    """Test for :func:`~validate_matrix_sum` using NumPy ndarray."""

    @pytest.mark.parametrize(
        "transition_matrix",
        (
            pytest.param(numpy_ndarray_valid(), id="type=numpy.ndarray"),
            pytest.param(scipy_coo_array_valid(), id="type=scipy.coo_array"),
            pytest.param(scipy_csc_array_valid(), id="type=scipy.csc_array"),
            pytest.param(scipy_csr_array_valid(), id="type=scipy.csr_array"),
        ),
    )
    def test_sum_to_one(self, transition_matrix):
        """Test if all rows sum to exactly one."""
        assert validate_matrix_sum(transition_matrix)

    @pytest.mark.parametrize(
        "transition_matrix",
        (
            pytest.param(numpy_ndarray_zero(), id="type=numpy.ndarray"),
            pytest.param(scipy_coo_array_zero(), id="type=scipy.coo_array"),
            pytest.param(scipy_csc_array_zeroes(), id="type=scipy.csc_array"),
            pytest.param(scipy_csr_array_zero(), id="type=scipy.csr_array"),
        ),
    )
    def test_all_zero(self, transition_matrix):
        """Test when all rows sum to exactly zero."""
        assert not validate_matrix_sum(transition_matrix)

    @pytest.mark.parametrize(
        "transition_matrix",
        (
            pytest.param(numpy_ndarray_negative(), id="type=numpy.ndarray"),
            pytest.param(
                scipy_coo_array_negative(), id="type=scipy.coo_array"
            ),
            pytest.param(
                scipy_csc_array_negative(), id="type=scipy.csc_array"
            ),
            pytest.param(
                scipy_csr_array_negative(), id="type=scipy.csr_array"
            ),
        ),
    )
    def test_negative(self, transition_matrix) -> None:
        """Test when all rows sum to exactly one but with negative numbers."""
        assert not validate_matrix_sum(transition_matrix)

    def test_logging_message(self, caplog: LogCaptureFixture):
        """Test whether a warning message is emitted."""
        array = np.array([[0, 1, 0], [1, 0, 0], [0, 0.5, 1]])
        with pytest.warns(TransitionMatrixSumWarning) as record:
            validate_matrix_sum(array)
        assert "Row 2 sums to 1.5" in str(record[0].message)


class TestTransitionMatrixPositive:
    """Tests for :func:`~validate_matrix_positive`."""

    @pytest.mark.parametrize(
        "transition_matrix",
        (
            pytest.param(numpy_ndarray_valid(), id="type=numpy.ndarray"),
            pytest.param(scipy_coo_array_valid(), id="type=scipy.coo_array"),
            pytest.param(scipy_csc_array_valid(), id="type=scipy.csc_array"),
            pytest.param(scipy_csr_array_valid(), id="type=scipy.csr_array"),
        ),
    )
    def test_all_positive(self, transition_matrix):
        """Test when all numbers are positive."""
        assert validate_matrix_negative(transition_matrix)

    @pytest.mark.parametrize(
        "transition_matrix",
        (
            pytest.param(numpy_ndarray_negative(), id="type=numpy.ndarray"),
            pytest.param(
                scipy_coo_array_negative(), id="type=scipy.coo_array"
            ),
            pytest.param(
                scipy_csc_array_negative(), id="type=scipy.csc_array"
            ),
            pytest.param(
                scipy_csr_array_negative(), id="type=scipy.csr_array"
            ),
        ),
    )
    def test_negative(self, transition_matrix) -> None:
        """Test when three rows contain a single negative number."""
        assert not validate_matrix_negative(transition_matrix)

    def test_logging_message(self, caplog: LogCaptureFixture) -> None:
        """Test whether a warning message is emitted.

        :param caplog: Object with logs captured from console.
        :type caplog: LogCaptureFixture
        """
        array = csr_array([[0, -1, 1], [1, 0, 0], [0, 0, 1]])
        with pytest.warns(TransitionMatrixNegativeWarning) as record:
            validate_matrix_negative(array)
        assert "index [0 1]" in str(record[0].message)
