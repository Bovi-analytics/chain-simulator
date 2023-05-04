import math

import cupyx.scipy.sparse as cps
from cupyx.profiler import benchmark
from cupy import array
import numpy as np


def generate_sparse_data(axis_size: int, cells_to_fill: int):
    rng = np.random.default_rng(1)
    if cells_to_fill:
        array_size = int(pow(axis_size, 2))
        cells = rng.choice(array_size, size=cells_to_fill, replace=False)
        rows, cols = np.divmod(cells, axis_size)
        data = rng.random(cells_to_fill)
    else:
        rows, cols, data = [], [], []
    return rows, cols, data


def construct_scipy_coo(rows, cols, data, shape):
    return cps.coo_matrix((data, (rows, cols)), shape=shape)


if __name__ == "__main__":
    print("Generating data")
    data = generate_sparse_data(8_388_608, math.floor(8_388_608))
    new_data = tuple(map(array, data))
    print("Making COO matrix")
    arr = construct_scipy_coo(*new_data, (8_388_608, 8_388_608))
    # arr.dot(arr)
    print("Making CSR matrix")
    # arr = arr.tocsr()
    new_arr = arr.copy()
    print(new_arr.shape)
    for _ in range(10_000 + 1):
        new_arr = new_arr.dot(arr)
    print("Done!")
    # bench = benchmark(construct_scipy_coo, data, n_repeat=5)
    # print(bench)
