"""Tests for module :mod:`~_simulation`."""

from itertools import zip_longest

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import sparse

try:
    import cupyx as cpx
except ImportError:
    cpx = None

from chain_simulator._simulation import ArrayProcessor, chain_simulator


@pytest.fixture
def numpy_array() -> NDArray[np.float32]:
    """Prepare a NumPy array.

    Returns
    -------
    numpy.ndarray
        NumPy array usable for testing purposes.
    """
    return np.array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])


class TestNumPy:
    """Tests for :func:`~chain_simulator` everything related to NumPy."""

    multiply_one = np.array([[0, 1 / 2, 1 / 2], [0, 1 / 4, 3 / 4], [0, 0, 1]])
    multiply_two = np.array([[0, 1 / 4, 3 / 4], [0, 1 / 8, 7 / 8], [0, 0, 1]])
    multiply_three = np.array(
        [[0, 1 / 8, 7 / 8], [0, 1 / 16, 15 / 16], [0, 0, 1]]
    )
    multiply_four = np.array(
        [[0, 1 / 16, 15 / 16], [0, 1 / 32, 31 / 32], [0, 0, 1]]
    )
    multiply_five = np.array(
        [[0, 1 / 32, 31 / 32], [0, 1 / 64, 63 / 64], [0, 0, 1]]
    )

    def test_multiply_negative_five(self, numpy_array) -> None:
        """Test `step` parameter of -5."""
        with pytest.raises(ValueError) as exc_info:
            next(chain_simulator(numpy_array, -5))
        assert "higher than 0" in exc_info.value.args[0]

    def test_multiply_zero(self, numpy_array) -> None:
        """Test `step` parameter of 0."""
        with pytest.raises(ValueError) as exc_info:
            next(chain_simulator(numpy_array, 0))
        assert "higher than 0" in exc_info.value.args[0]

    def test_multiply_one(self, numpy_array) -> None:
        """Test `step` parameter of 1."""
        result = next(chain_simulator(numpy_array, 1))
        assert np.all(result[0] == self.multiply_one)
        assert result[1] == 1

    def test_multiply_two(self, numpy_array) -> None:
        """Test `step` parameter of 2."""
        result = next(chain_simulator(numpy_array, 2))
        assert np.all(result[0] == self.multiply_two)
        assert result[1] == 2

    def test_multiply_five(self, numpy_array) -> None:
        """Test `step` parameter of 5."""
        result = next(chain_simulator(numpy_array, 5))
        assert np.all(result[0] == self.multiply_five)
        assert result[1] == 5

    def test_intermediary_negative(self, numpy_array) -> None:
        """Test `interval` parameter of -1."""
        with pytest.raises(ValueError):
            tuple(chain_simulator(numpy_array, 2, -1))

    def test_intermediary_none(self, numpy_array) -> None:
        """Test `interval` parameter of 0."""
        results = tuple(chain_simulator(numpy_array, 3, 0))
        expected = ((self.multiply_three, 3),)
        assert len(results) == 1
        assert np.all(results[0][0] == expected[0][0])
        assert results[0][1] == expected[0][1]

    def test_intermediary_all(self, numpy_array) -> None:
        """Test `interval` parameter of 1."""
        arrays_actual, steps_actual = zip(*chain_simulator(numpy_array, 3, 1))
        arrays_expected, steps_expected = zip(
            *(
                (self.multiply_one, 1),
                (self.multiply_two, 2),
                (self.multiply_three, 3),
            )
        )
        comparisons = [
            np.any(actual == expected)
            for actual, expected in zip_longest(arrays_actual, arrays_expected)
        ]
        assert len(arrays_actual) == 3
        assert all(comparisons)
        assert steps_actual == steps_expected

    def test_intermediary_every_second(self, numpy_array) -> None:
        """Test `interval` parameter of 2."""
        arrays_actual, steps_actual = zip(*chain_simulator(numpy_array, 5, 2))
        arrays_expected, steps_expected = zip(
            *(
                (self.multiply_two, 2),
                (self.multiply_four, 4),
                (self.multiply_five, 5),
            )
        )
        comparisons = [
            np.any(actual == expected)
            for actual, expected in zip_longest(arrays_actual, arrays_expected)
        ]
        assert len(arrays_actual) == 3
        assert all(comparisons)
        assert steps_actual == steps_expected

    def test_intermediary_every_third(self, numpy_array) -> None:
        """Test `interval` parameter of 3."""
        arrays_actual, steps_actual = zip(*chain_simulator(numpy_array, 5, 3))
        arrays_expected, steps_expected = zip(
            *(
                (self.multiply_three, 3),
                (self.multiply_five, 5),
            )
        )
        comparisons = [
            np.any(actual == expected)
            for actual, expected in zip_longest(arrays_actual, arrays_expected)
        ]
        assert len(arrays_actual) == 2
        assert all(comparisons)
        assert steps_actual == steps_expected


class TestArrayProcessor:
    numpy_initial_state_vector = np.array([1, 0, 0])
    numpy_final_state_vector = np.array([0, 0.5, 0.5])

    scipy_sparse_supported = (
        sparse.coo_array,
        sparse.coo_matrix,
        sparse.csc_array,
        sparse.csc_matrix,
        sparse.csr_array,
        sparse.csr_matrix,
    )

    def test_numpy_ndarray(self, numpy_array):
        result = next(
            ArrayProcessor(numpy_array, self.numpy_initial_state_vector, 1)
        )
        assert np.all(result[0] == self.numpy_final_state_vector)

    @pytest.mark.parametrize("sparse_format", scipy_sparse_supported)
    def test_scipy_csc_array(self, sparse_format, numpy_array):
        sparse_matrix = sparse_format(numpy_array)
        result = next(
            ArrayProcessor(sparse_matrix, self.numpy_initial_state_vector, 1)
        )
        assert np.all(result[0] == self.numpy_final_state_vector)
