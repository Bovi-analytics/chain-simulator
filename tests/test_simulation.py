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


class TestChainSimulator:
    """Tests for :func:`~chain_simulator` everything related to NumPy."""

    initial_matrix = np.array([[0, 1, 0], [0, 1 / 2, 1 / 2], [0, 0, 1]])

    multiply_zero = np.array([[0, 1, 0], [0, 1 / 2, 1 / 2], [0, 0, 1]])
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
    multiply_six = np.array(
        [[0, 1 / 64, 63 / 64], [0, 1 / 128, 127 / 128], [0, 0, 1]]
    )
    multiply_seven = np.array(
        [[0, 1 / 128, 127 / 128], [0, 1 / 256, 255 / 256], [0, 0, 1]]
    )
    multiply_eight = np.array(
        [[0, 1 / 256, 255 / 256], [0, 1 / 512, 511 / 512], [0, 0, 1]]
    )
    multiply_nine = np.array(
        [[0, 1 / 512, 511 / 512], [0, 1 / 1024, 1023 / 1024], [0, 0, 1]]
    )

    @pytest.mark.parametrize(
        "matrix, steps",
        (
            pytest.param(initial_matrix, -10, id="steps=-10"),
            pytest.param(initial_matrix, -5, id="steps=-5"),
            pytest.param(initial_matrix, 0, id="steps=0"),
        ),
    )
    def test_fail_steps_below_1(self, matrix, steps):
        """Test `step` parameter with numbers below 1."""
        with pytest.raises(ValueError) as exc_info:
            next(chain_simulator(matrix, steps))
        assert "higher than 0" in exc_info.value.args[0]

    @pytest.mark.parametrize(
        "matrix, steps, interval",
        (
            pytest.param(initial_matrix, 1, -10, id="interval=-10"),
            pytest.param(initial_matrix, 1, -5, id="interval=-5"),
            pytest.param(initial_matrix, 1, -1, id="interval=-1"),
        ),
    )
    def test_fail_interval_below_1(self, matrix, steps, interval):
        """Test `interval` parameter with values below 0."""
        with pytest.raises(ValueError):
            tuple(chain_simulator(numpy_array, 2, -1))

    @pytest.mark.parametrize(
        "matrix, steps, matrix_expected",
        (
            pytest.param(initial_matrix, 1, multiply_zero, id="steps=1"),
            pytest.param(initial_matrix, 2, multiply_one, id="steps=2"),
            pytest.param(initial_matrix, 5, multiply_four, id="steps=5"),
        ),
    )
    def test_correct_output_matrix(self, matrix, steps, matrix_expected):
        """Test whether a correct final transition matrix is returned."""
        result = next(chain_simulator(matrix, steps))
        assert np.all(result[0] == matrix_expected)

    @pytest.mark.parametrize(
        "matrix, steps, step_expected",
        (
            pytest.param(initial_matrix, 1, 1, id="steps=1"),
            pytest.param(initial_matrix, 2, 2, id="steps=2"),
            pytest.param(initial_matrix, 5, 5, id="steps=5"),
        ),
    )
    def test_correct_output_step(self, matrix, steps, step_expected):
        """Test whether a correct final step is returned."""
        result = next(chain_simulator(matrix, steps))
        assert result[1] == step_expected

    @pytest.mark.parametrize(
        "matrix, steps, interval, steps_expected",
        (
            pytest.param(initial_matrix, 10, 0, (10,), id="interval=0"),
            pytest.param(
                initial_matrix,
                10,
                1,
                (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                id="interval=1",
            ),
            pytest.param(
                initial_matrix, 10, 2, (2, 4, 6, 8, 10), id="interval=2"
            ),
            pytest.param(
                initial_matrix, 10, 3, (3, 6, 9, 10), id="interval=3"
            ),
            pytest.param(initial_matrix, 10, 4, (4, 8, 10), id="interval=4"),
            pytest.param(initial_matrix, 10, 5, (5, 10), id="interval=5"),
            pytest.param(initial_matrix, 10, 6, (6, 10), id="interval=6"),
            pytest.param(initial_matrix, 10, 7, (7, 10), id="interval=7"),
            pytest.param(initial_matrix, 10, 8, (8, 10), id="interval=8"),
            pytest.param(initial_matrix, 10, 9, (9, 10), id="interval=9"),
            pytest.param(initial_matrix, 10, 10, (10,), id="interval=10"),
            pytest.param(initial_matrix, 10, 11, (10,), id="interval=11"),
        ),
    )
    def test_intermediary_steps(self, matrix, steps, interval, steps_expected):
        """Test whether intermediary steps are correct."""
        _, steps_returned = zip(
            *tuple(chain_simulator(matrix, steps, interval))
        )
        assert steps_returned == steps_expected

    @pytest.mark.parametrize(
        "matrix, steps, interval, matrices_expected",
        (
            pytest.param(
                initial_matrix, 10, 0, (multiply_nine,), id="interval=0"
            ),
            pytest.param(
                initial_matrix,
                10,
                1,
                (
                    multiply_zero,
                    multiply_one,
                    multiply_two,
                    multiply_three,
                    multiply_four,
                    multiply_five,
                    multiply_six,
                    multiply_seven,
                    multiply_eight,
                    multiply_nine,
                ),
                id="interval=1",
            ),
            pytest.param(
                initial_matrix,
                10,
                2,
                (
                    multiply_one,
                    multiply_three,
                    multiply_five,
                    multiply_seven,
                    multiply_nine,
                ),
                id="interval=2",
            ),
            pytest.param(
                initial_matrix,
                10,
                3,
                (
                    multiply_two,
                    multiply_five,
                    multiply_eight,
                    multiply_nine,
                ),
                id="interval=3",
            ),
            pytest.param(
                initial_matrix,
                10,
                4,
                (
                    multiply_three,
                    multiply_seven,
                    multiply_nine,
                ),
                id="interval=4",
            ),
            pytest.param(
                initial_matrix,
                10,
                5,
                (
                    multiply_four,
                    multiply_nine,
                ),
                id="interval=5",
            ),
            pytest.param(
                initial_matrix,
                10,
                6,
                (
                    multiply_five,
                    multiply_nine,
                ),
                id="interval=6",
            ),
            pytest.param(
                initial_matrix,
                10,
                7,
                (
                    multiply_six,
                    multiply_nine,
                ),
                id="interval=7",
            ),
            pytest.param(
                initial_matrix,
                10,
                8,
                (
                    multiply_seven,
                    multiply_nine,
                ),
                id="interval=8",
            ),
            pytest.param(
                initial_matrix,
                10,
                9,
                (
                    multiply_eight,
                    multiply_nine,
                ),
                id="interval=9",
            ),
            pytest.param(
                initial_matrix, 10, 10, (multiply_nine,), id="interval=10"
            ),
            pytest.param(
                initial_matrix, 10, 11, (multiply_nine,), id="interval=11"
            ),
        ),
    )
    def test_intermediary_matrix(
        self, matrix, steps, interval, matrices_expected
    ):
        """Test whether intermediary matrices are correct."""
        matrices_returned, _ = zip(
            *tuple(chain_simulator(matrix, steps, interval))
        )
        combinations = zip_longest(matrices_returned, matrices_expected)
        assert all(
            np.all(returned == expected) for returned, expected in combinations
        )


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
        pytest.param(
            as_cupy_ndarray(numpy_initial_vector),
            as_cupy_ndarray(numpy_matrix),
            id="cupy.ndarray cupy.ndarray",
        ),
        pytest.param(
            as_cupy_ndarray(numpy_initial_vector),
            as_cupyx_csc_matrix(sparse.csc_matrix(numpy_matrix)),
            id="cupy.ndarray cupyx.csc_matrix",
        ),
        pytest.param(
            as_cupy_ndarray(numpy_initial_vector),
            as_cupyx_csr_matrix(sparse.csr_matrix(numpy_matrix)),
            id="cupy.ndarray cupyx.csr_matrix",
        ),
    )

    @pytest.mark.parametrize("vec_initial, matrix", cupy_formats)
    def test_cupy_supported_type(self, vec_initial, matrix):
        results = next(self.partial_processor(vec_initial, matrix))
        assert isinstance(results[0], _cupy.ndarray)


class TestStateVectorProcessor:
    processor_final = partial(state_vector_processor, steps=3)
    processor_intermediate_all = partial(processor_final, steps=3, interval=1)
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
