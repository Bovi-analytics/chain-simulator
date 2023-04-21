import sys
from timeit import Timer

from numpy import arange, dtype
from numpy.random import default_rng
from scipy.sparse import csr_array, random
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from benchmarks.abstract import AbstractArrayInfo, csv_writer


class CSRArrayInfo(AbstractArrayInfo[csr_array]):
    @classmethod
    def shape(cls: Self, array: csr_array) -> tuple[int, ...]:
        return array.shape

    @classmethod
    def size(cls: Self, array: csr_array) -> int:
        return array.size

    @classmethod
    def nbytes(cls: Self, array: csr_array) -> int:
        return array.data.nbytes

    @classmethod
    def itemsize(cls: Self, array: csr_array) -> int:
        return array.data.itemsize

    @classmethod
    def dtype(cls: Self, array: csr_array) -> dtype:
        return array.data.dtype

    @classmethod
    def count_nonzero(cls: Self, array: csr_array) -> int:
        return array.nnz


def benchmark_sparsity_2d():
    shape = 2048
    densities = arange(0, 1 + 0.1, 0.1)
    setup = "import scipy.sparse as cs; import numpy as np; array = cs.csr_array(cs.random({shape}, {shape}, density={density}, format='csr', dtype='float64', random_state=np.random.default_rng(1)))"
    stmt = "array.dot(array)"
    data = []
    for density in densities:
        timer = Timer(stmt, setup.format(density=density, shape=shape))
        timings = timer.repeat(repeat=10, number=10)
        array = csr_array(
            random(
                shape,
                shape,
                density=density,
                format="csr",
                dtype="float64",
                random_state=default_rng(1),
            )
        )
        data.append(*CSRArrayInfo.as_tuple(array), *timings)
    runs = [f"run_{run:02}" for run in range(1, 10 + 1)]
    header = [*CSRArrayInfo.header(), *runs]
    csv_writer("benchmark_scipy_sparsity_2d", header, data)


if __name__ == "__main__":
    benchmark_sparsity_2d()
