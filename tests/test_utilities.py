"""Tests for module :mod:`~chain_simulator.utilities`."""

import numpy as np
import numpy.typing as npt
from _pytest.logging import LogCaptureFixture
from chain_simulator.utilities import (
    validate_matrix_positive,
    validate_matrix_sum,
)
from scipy.sparse import coo_array, csr_array
from typing_extensions import Self

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

    def test_sum_to_one(self: Self) -> None:
        """Test if all rows sum to exactly one."""
        assert validate_matrix_sum(numpy_ndarray_valid())
        assert validate_matrix_sum(scipy_coo_array_valid())
        assert validate_matrix_sum(scipy_csr_array_valid())

    def test_all_zero(self: Self) -> None:
        """Test when all rows sum to exactly zero."""
        assert not validate_matrix_sum(numpy_ndarray_zero())
        assert not validate_matrix_sum(scipy_coo_array_zero())
        assert not validate_matrix_sum(scipy_csr_array_zero())

    def test_negative(self: Self) -> None:
        """Test when all rows sum to exactly one but with negative numbers."""
        assert not validate_matrix_sum(numpy_ndarray_negative())
        assert not validate_matrix_sum(scipy_coo_array_negative())
        assert not validate_matrix_sum(scipy_csr_array_negative())

    def test_logging_message(self: Self, caplog: LogCaptureFixture) -> None:
        """Test whether a warning message is emitted."""
        array = np.array([[0, 1, 0], [1, 0, 0], [0, 0.5, 1]])
        validate_matrix_sum(array)
        for record in caplog.records[:1]:
            assert record.levelname == "WARNING"
            assert "Row 2" in record.message
            assert "1.5" in record.message


class TestTransitionMatrixPositive:
    """Tests for :func:`~validate_matrix_positive`."""

    def test_all_positive(self: Self) -> None:
        """Test when all numbers are positive."""
        array = csr_array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
        assert validate_matrix_positive(array)

    def test_one_negative(self: Self) -> None:
        """Test when one row contains a single negative number."""
        array = csr_array([[-1, 1, 0], [1, 0, 0], [0, 0, 1]])
        assert not validate_matrix_positive(array)

    def test_three_negative(self: Self) -> None:
        """Test when three rows all contain negative numbers."""
        array = csr_array([[-1, 1, 0], [-1, 0, 0], [0, 0, -1]])
        assert not validate_matrix_positive(array)

    def test_numpy_array(self: Self) -> None:
        """Test when there is a faulty NumPy array."""
        array = np.array([[-1, 1, 0], [-1, 0, 0], [0, 0, -1]])
        assert not validate_matrix_positive(array)

    def test_numpy_valid(self: Self) -> None:
        """Test when there is a valid NumPy array."""
        array = np.array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
        assert validate_matrix_positive(array)

    def test_logging_message(self: Self, caplog: LogCaptureFixture) -> None:
        """Test whether a warning message is emitted.

        :param caplog: Object with logs captured from console.
        :type caplog: LogCaptureFixture
        """
        array = csr_array([[0, -1, 1], [1, 0, 0], [0, 0, 1]])
        validate_matrix_positive(array)
        for record in caplog.records[:1]:
            assert record.levelname == "WARNING"
            assert "index [0 1]" in record.message
            assert "Probability -1" in record.message
