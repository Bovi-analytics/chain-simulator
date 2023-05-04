from timeit import Timer

from numpy import arange, dtype
from numpy.random import default_rng
from scipy.sparse import csr_array, random, bsr_array
from typing_extensions import Self

from benchmarks.abstract import AbstractArrayInfo, _T, csv_writer, timert


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

    @classmethod
    def format(cls: Self, array: csr_array) -> float:
        return array.format

    # @classmethod
    # def as_tuple(cls, array: _T) -> tuple[
    #     tuple[int, ...], int, int, int, dtype, int, str]:
    #     return *super(array).as_tuple(array), cls.format(array)
    #
    # @staticmethod
    # def header() -> tuple[str, ...]:
    #     return *super().header(), "format"


def benchmark_sparsity_2d():
    shape = 2048
    densities = arange(0, 1 + 0.1, 0.1)
    # setup = "import scipy.sparse as cs; import numpy as np; array = cs.csr_array(cs.random({shape}, {shape}, density={density}, format='csr', dtype='float64', random_state=np.random.default_rng(1)))"
    stmt = "array.dot(array)"
    data = []
    for density in densities:
        timings = timert(operation, assemble_sparse_array)
        # timer = Timer(stmt, setup.format(density=density, shape=shape))
        # timings = timer.repeat(repeat=10, number=10)
        # array = csr_array(
        #     random(
        #         shape,
        #         shape,
        #         density=density,
        #         format="csr",
        #         dtype="float64",
        #         random_state=default_rng(1),
        #     )
        # )
        array = assemble_sparse_array()
        data.append((*CSRArrayInfo.as_tuple(array), *timings))
    runs = [f"run_{run:02}" for run in range(1, 10 + 1)]
    header = [*CSRArrayInfo.header(), *runs]
    csv_writer("benchmark_scipy_sparsity_2d", header, data)


def assemble_sparse_array(
    shape: tuple[int, int] = (20, 20),
    density: float = 1,
    format: str = "csr",
    dtype: str = "float64",
) -> csr_array:
    array = random(
        *shape,
        density=density,
        format=format,
        dtype=dtype,
        random_state=default_rng(1),
    )
    return csr_array(array)


def operation(array):
    array.dot(array)


if __name__ == "__main__":
    benchmark_sparsity_2d()
