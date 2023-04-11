"""Tests for module :mod:`~chain_simulator.utilities.`."""
import numpy as np
from chain_simulator.utilities import (
    validate_matrix_positive,
    validate_matrix_sum,
)
from scipy.sparse import csr_array
from typing_extensions import Self


class TestTransitionMatrixSum:
    """Tests for :func:`~chain_simulator.utilities.validate_matrix_sum`."""

    def test_sum_to_one(self: Self) -> None:
        """Test if all rows sum to exactly one."""
        array = csr_array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
        assert validate_matrix_sum(array)

    def test_all_zero(self: Self) -> None:
        """Test when all rows sum to exactly zero."""
        array = csr_array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        result = validate_matrix_sum(array)
        assert not result

    def test_negative(self: Self) -> None:
        """Test when all rows sum to exactly one but with negative numbers."""
        array = csr_array([[-1, 1, 0], [1, 0, -1], [0, -1, 1]])
        assert not validate_matrix_sum(array)

    def test_numpy_array(self: Self) -> None:
        """Test when all rows sum to exactly one but with negative numbers."""
        array = np.array([[-1, 1, 0], [1, 0, -1], [0, -1, 1]])
        assert not validate_matrix_sum(array)


class TestTransitionMatrixPositive:
    """Tests for :func:`~chain_simulator.utilities.validate_matrix_positive`."""

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

    def test_numpy_array(self):
        array = np.array([[-1, 1, 0], [-1, 0, 0], [0, 0, -1]])
        assert not validate_matrix_positive(array)

    def test_numpy_valid(self):
        array = np.array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
        assert validate_matrix_positive(array)
