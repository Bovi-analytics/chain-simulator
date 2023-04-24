import sys
from timeit import Timer
from typing import Any

from numpy import arange, count_nonzero, dtype
from numpy.random import default_rng
from numpy.typing import NDArray

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from benchmarks.abstract import (
    _T,
    AbstractArrayInfo,
    csv_writer,
)


class ArrayInfo(AbstractArrayInfo[NDArray[Any]]):
    @classmethod
    def shape(cls: Self, array: _T) -> tuple[int, ...]:
        return array.shape

    @classmethod
    def size(cls: Self, array: _T) -> int:
        return array.size

    @classmethod
    def nbytes(cls: Self, array: _T) -> int:
        return array.nbytes

    @classmethod
    def itemsize(cls: Self, array: _T) -> int:
        return array.itemsize

    @classmethod
    def dtype(cls: Self, array: _T) -> dtype:
        return array.dtype

    @classmethod
    def count_nonzero(cls: Self, array: _T) -> int:
        return count_nonzero(array)


def benchmark_size_2d():
    shapes = range(1024, 4096 + 1, 64)
    iterations = 20
    repeats = 10
    setup = "import numpy as np; array = np.random.default_rng(1).random(({shape}, {shape}), 'float64')"
    stmt = "array.dot(array)"
    data = []
    for shape in shapes:
        timer = Timer(stmt, setup.format(shape=shape))
        timings = timer.repeat(number=iterations, repeat=repeats)
        array = default_rng(1).random((shape, shape), "float64")
        data.append((*ArrayInfo.as_tuple(array), *timings))
        print(f"Shape {shape} of {shapes[-1]}")
    head = [f"run_{run:02}" for run in range(1, repeats + 1)]
    header = [*ArrayInfo.header(), *head]
    csv_writer("benchmark_numpy_size_2d", header, data)


def benchmark_sparsity_2d():
    shape = 2048
    sparsity = arange(0, 1, 0.1)
    setup = "import numpy as np; array = np.random.default_rng(1).random(({shape}, {shape}), 'float64'); array[array < {sparse}] = 0"
    stmt = "array.dot(array)"
    data = []
    for sparse in sparsity:
        timer = Timer(stmt, setup.format(shape=shape, sparse=sparse))
        timings = timer.repeat(repeat=10, number=20)
        array = default_rng(1).random((shape, shape), "float64")
        array[array < sparse] = 0
        data.append((*ArrayInfo.as_tuple(array), *timings))
        print(f"Sparsity {sparse} of {sparsity[-1]}")
    runs = [f"run_{run:02}" for run in range(1, 10 + 1)]
    header = [*ArrayInfo.header(), *runs]
    csv_writer("benchmark_numpy_sparsity_2d", header, data)


if __name__ == "__main__":
    benchmark_size_2d()
    benchmark_sparsity_2d()
