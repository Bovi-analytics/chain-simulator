import sys
from itertools import zip_longest

import numpy as np
from pytest import fixture
from chain_simulator import chain_simulator

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


@fixture
def numpy_array():
    return np.array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])


class TestNumPy:
    multiply_one = np.array(
        [[0, 1 / 2, 1 / 2], [0, 1 / 4, 3 / 4], [0, 0, 1]]
    )
    multiply_two = np.array(
        [[0, 1 / 4, 3 / 4], [0, 1 / 8, 7 / 8], [0, 0, 1]]
    )
    multiply_three = np.array(
        [[0, 1 / 8, 7 / 8], [0, 1 / 16, 15 / 16], [0, 0, 1]]
    )
    multiply_four = np.array(
        [[0, 1 / 16, 15 / 16], [0, 1 / 32, 31 / 32], [0, 0, 1]]
    )
    multiply_five = np.array(
        [[0, 1 / 32, 31 / 32], [0, 1 / 64, 63 / 64], [0, 0, 1]]
    )

    def test_multiply_one(self: Self, numpy_array) -> None:
        result = next(chain_simulator(numpy_array, 1))
        assert np.all(result[0] == self.multiply_one)
        assert result[1] == 1

    def test_multiply_two(self: Self, numpy_array) -> None:
        result = next(chain_simulator(numpy_array, 2))
        assert np.all(result[0] == self.multiply_two)
        assert result[1] == 2

    def test_multiply_five(self: Self, numpy_array) -> None:
        result = next(chain_simulator(numpy_array, 5))
        assert np.all(result[0] == self.multiply_five)
        assert result[1] == 5

    def test_intermediary_none(self, numpy_array):
        results = tuple(chain_simulator(numpy_array, 3, 0))
        expected = ((self.multiply_three, 3),)
        assert len(results) == 1
        assert np.all(results[0][0] == expected[0][0])
        assert results[0][1] == expected[0][1]

    def test_intermediary_all(self, numpy_array):
        arrays_actual, steps_actual = zip(*chain_simulator(numpy_array, 3, 1))
        arrays_expected, steps_expected = zip(*(
            (self.multiply_one, 1),
            (self.multiply_two, 2),
            (self.multiply_three, 3),
        ))
        comparisons = [np.any(actual == expected) for actual, expected in
                       zip_longest(arrays_actual, arrays_expected)]
        assert len(arrays_actual) == 3
        assert all(comparisons)
        assert steps_actual == steps_expected

    def test_intermediary_every_second(self, numpy_array):
        arrays_actual, steps_actual = zip(*chain_simulator(numpy_array, 5, 2))
        arrays_expected, steps_expected = zip(*(
            (self.multiply_two, 2),
            (self.multiply_four, 4),
            (self.multiply_five, 5),
        ))
        comparisons = [np.any(actual == expected) for actual, expected in
                       zip_longest(arrays_actual, arrays_expected)]
        assert len(arrays_actual) == 3
        assert all(comparisons)
        assert steps_actual == steps_expected

    def test_intermediary_every_third(self, numpy_array):
        arrays_actual, steps_actual = zip(*chain_simulator(numpy_array, 5, 3))
        arrays_expected, steps_expected = zip(*(
            (self.multiply_three, 3),
            (self.multiply_five, 5),
        ))
        comparisons = [np.any(actual == expected) for actual, expected in
                       zip_longest(arrays_actual, arrays_expected)]
        assert len(arrays_actual) == 2
        assert all(comparisons)
        assert steps_actual == steps_expected

    # def test_intermediary_everything(self: Self, numpy_array) -> None:
    #     results = tuple(chain_simulator(numpy_array, 2, 1))

# class TestChainSimulator:
#     """Tests for :func:`~chain_simulator.implementations.chain_simulator`."""
#
#     def test_matmul_1(self: Self) -> None:
#         """Test matrix multiplication once."""
#         array = coo_array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
#         result = chain_simulator(array, 1)
#         expected = coo_array(
#             [[0.00, 0.50, 0.50], [0.00, 0.25, 0.75], [0.00, 0.00, 1.00]]
#         )
#         comparison = result != expected
#         assert comparison.size <= 0
#
#     def test_matmul_2(self: Self) -> None:
#         """Test matrix multiplication twice."""
#         array = coo_array([[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
#         result = chain_simulator(array, 2)
#         expected = coo_array(
#             [
#                 [0.000, 0.250, 0.750],
#                 [0.000, 0.125, 0.875],
#                 [0.000, 0.000, 1.000],
#             ]
#         )
#         comparison = result != expected
#         assert comparison.size <= 0
