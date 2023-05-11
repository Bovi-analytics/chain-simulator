"""Tests for module :mod:`~_simulation`."""
from functools import partial
from itertools import zip_longest

import numpy as np
import pytest
from numpy.typing import NDArray
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

from chain_simulator._simulation import (
    chain_simulator,
    state_vector_processor,
    vector_processor_cupy,
    vector_processor_numpy,
    vector_processor_scipy,
)

SKIP_CUPY_ABSENT = pytest.mark.skipif(
    not _cupy_installed, reason="CuPy is not installed"
)


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


class TestStateVectorProcessorNumPy:
    partial_processor = partial(vector_processor_numpy, steps=3)

    numpy_initial_vector = np.array([1, 0, 0])
    numpy_matrix = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]
    )

    numpy_formats = ((numpy_initial_vector, numpy_matrix),)

    @pytest.mark.parametrize("vec_initial, matrix", numpy_formats)
    def test_supported_type(self, vec_initial, matrix):
        results = next(self.partial_processor(vec_initial, matrix))
        assert isinstance(results[0], np.ndarray)


class TestStateVectorProcessorSciPy:
    partial_processor = partial(vector_processor_scipy, steps=3)
    numpy_initial_vector = np.array([1, 0, 0])
    numpy_matrix = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]
    )

    scipy_formats = (
        (numpy_initial_vector, sparse.csc_array(numpy_matrix)),
        (numpy_initial_vector, sparse.csc_matrix(numpy_matrix)),
        (numpy_initial_vector, sparse.csr_array(numpy_matrix)),
        (numpy_initial_vector, sparse.csr_matrix(numpy_matrix)),
    )

    @pytest.mark.parametrize("vec_initial, matrix", scipy_formats)
    def test_scipy_supported_type(self, vec_initial, matrix):
        results = next(self.partial_processor(vec_initial, matrix))
        assert isinstance(results[0], np.ndarray)


def as_cupy_ndarray(array):
    if _cupy_installed:
        return _cupy.array(array)
    return None


def as_cupyx_csc_matrix(array):
    if _cupy_installed:
        return _cupyx.scipy.sparse.csc_matrix(array)
    return None


def as_cupyx_csr_matrix(array):
    if _cupy_installed:
        return _cupyx.scipy.sparse.csr_matrix(array)
    return None


@pytest.mark.gpu
@pytest.mark.skipif(not _cupy_installed, reason="CuPy is not installed")
class TestStateVectorProcessorCuPy:
    partial_processor = partial(vector_processor_cupy, steps=3)
    numpy_initial_vector = np.array([1, 0, 0])
    numpy_matrix = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]
    )

    cupy_formats = (
        (
            as_cupy_ndarray(numpy_initial_vector),
            as_cupy_ndarray(numpy_matrix),
        ),
        (
            as_cupy_ndarray(numpy_initial_vector),
            as_cupyx_csc_matrix(sparse.csc_matrix(numpy_matrix)),
        ),
        (
            as_cupy_ndarray(numpy_initial_vector),
            as_cupyx_csr_matrix(sparse.csr_matrix(numpy_matrix)),
        ),
    )

    @pytest.mark.parametrize("vec_initial, matrix", cupy_formats)
    def test_cupy_supported_type(self, vec_initial, matrix):
        results = next(self.partial_processor(vec_initial, matrix))
        assert isinstance(results[0], _cupy.ndarray)

    class TestStateVectorProcessor:
        processor_final = partial(state_vector_processor, steps=3)
        processor_intermediate_all = partial(
            processor_final, steps=3, interval=1
        )
        processor_intermediate_second = partial(
            processor_final, steps=3, interval=2
        )

        numpy_initial_vector = np.array([1, 0, 0])
        numpy_matrix = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]
        )

        valid_combinations = (
            pytest.param(
                numpy_initial_vector,
                numpy_matrix,
                id="numpy.ndarray numpy.ndarray",
            ),
            pytest.param(
                numpy_initial_vector,
                sparse.csc_array(numpy_matrix),
                id="numpy.ndarray scipy.csc_array",
            ),
            pytest.param(
                numpy_initial_vector,
                sparse.csc_matrix(numpy_matrix),
                id="numpy.ndarray scipy.csc_matrix",
            ),
            pytest.param(
                numpy_initial_vector,
                sparse.csr_array(numpy_matrix),
                id="numpy.ndarray scipy.csr_array",
            ),
            pytest.param(
                numpy_initial_vector,
                sparse.csr_matrix(numpy_matrix),
                id="numpy.ndarray scipy.csr_matrix",
            ),
            pytest.param(
                numpy_initial_vector,
                sparse.coo_array(numpy_matrix),
                id="numpy.ndarray scipy.coo_array",
            ),
            pytest.param(
                numpy_initial_vector,
                sparse.coo_matrix(numpy_matrix),
                id="numpy.ndarray scipy.coo_matrix",
            ),
            pytest.param(
                numpy_initial_vector,
                as_cupy_ndarray(numpy_matrix),
                id="numpy.ndarray cupy.ndarray",
                marks=SKIP_CUPY_ABSENT,
            ),
            pytest.param(
                numpy_initial_vector,
                as_cupyx_csc_matrix(sparse.csc_matrix(numpy_matrix)),
                id="numpy.ndarray cupyx.csc_matrix",
                marks=SKIP_CUPY_ABSENT,
            ),
            pytest.param(
                numpy_initial_vector,
                as_cupyx_csr_matrix(sparse.csr_matrix(numpy_matrix)),
                id="numpy.ndarray cupyx.csr_matrix",
                marks=SKIP_CUPY_ABSENT,
            ),
            pytest.param(
                as_cupy_ndarray(numpy_initial_vector),
                as_cupy_ndarray(numpy_matrix),
                id="cupy.ndarray cupy.ndarray",
                marks=SKIP_CUPY_ABSENT,
            ),
            pytest.param(
                as_cupy_ndarray(numpy_initial_vector),
                as_cupyx_csc_matrix(sparse.csc_matrix(numpy_matrix)),
                id="cupy.ndarray cupyx.csc_matrix",
                marks=SKIP_CUPY_ABSENT,
            ),
            pytest.param(
                as_cupy_ndarray(numpy_initial_vector),
                as_cupyx_csr_matrix(sparse.csr_matrix(numpy_matrix)),
                id="cupy.ndarray cupyx.csr_matrix",
                marks=SKIP_CUPY_ABSENT,
            ),
        )

        @pytest.mark.parametrize("vector, matrix", valid_combinations)
        def test_output_type(self, vector, matrix):
            results = next(self.processor_final(vector, matrix))
            assert isinstance(results[0], np.ndarray)
            assert isinstance(results[1], int)

        @pytest.mark.parametrize("vector, matrix", valid_combinations)
        def test_output_no_intermediate(self, vector, matrix):
            results = tuple(self.processor_final(vector, matrix))
            assert len(results) == 1

        @pytest.mark.parametrize("vector, matrix", valid_combinations)
        def test_output_all_intermediate(self, vector, matrix):
            results = tuple(self.processor_intermediate_all(vector, matrix))
            assert len(results) == 3
