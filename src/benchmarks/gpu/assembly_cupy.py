import scipy.sparse
from cupyx.profiler import benchmark
from cupyx.scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from math import pow

import numpy as np


# from scipy.sparse import


def prepare_data(axis_size, cells_to_fill):
    rng = np.random.default_rng(1)
    if cells_to_fill:
        array_size = int(pow(axis_size, 2))
        cells = rng.choice(array_size, size=cells_to_fill, replace=False)
        data = rng.random(cells_to_fill)
        rows, cols = np.divmod(cells, axis_size)
    else:
        data, rows, cols = [], [], []
    return data, rows, cols


def apples_to_apples():
    coo_data = prepare_data(2048, 2048 ** 2)
    csr_prep = scipy.sparse.coo_matrix(coo_data).tocsr()
    csr_data =




def cupy_dense():
    pass


def cupy_scipy_coo(data, rows, cols):
    return coo_matrix((data, (rows, cols)))


def cupy_scipy_csc(data, indices, indptr):
    return csc_matrix((data, indices, indptr))


def cupy_scipy_csr(data, indices, indptr):
    return csr_matrix((data, indices, indptr))


if __name__ == "__main__":
    pass
