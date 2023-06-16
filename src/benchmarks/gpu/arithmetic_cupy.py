import cupy
import cupyx.scipy.sparse
import numpy as np
import scipy
from cupyx.profiler import benchmark


def matrix_matrix_coo(matrix1, matrix2):
    return cupyx.scipy.sparse.coo_matrix.dot(matrix1, matrix2)


def matrix_matrix_csc(matrix1, matrix2):
    return cupyx.scipy.sparse.csc_matrix.dot(matrix1, matrix2)


def matrix_matrix_csr(matrix1, matrix2):
    return cupyx.scipy.sparse.csr_matrix.dot(matrix1, matrix2)


def vector_matrix_coo(vector, matrix):
    return cupyx.scipy.sparse.coo_matrix.dot(vector, matrix)


def vector_matrix_csc(vector, matrix):
    return cupyx.scipy.sparse.csc_matrix.dot(vector, matrix)


def vector_matrix_csr(vector, matrix):
    return cupyx.scipy.sparse.csr_matrix.dot(vector, matrix)


def generate_sparse_data(axis_size: int, cells_to_fill: int):
    rng = np.random.default_rng(1)
    if cells_to_fill:
        array_size = int(pow(axis_size, 2))
        cells = rng.choice(array_size, size=cells_to_fill, replace=False)
        data = rng.random(cells_to_fill)
        rows, cols = np.divmod(cells, axis_size)
    else:
        data, rows, cols = [], [], []
    return data, rows, cols


def setup(nodes: int, edges: int):
    data = generate_sparse_data(nodes, edges)
    repeats = 5_000
    scipy_matrix = scipy.sparse.coo_array((data[0], (data[1], data[2])), shape=(nodes, nodes))
    cupy_coo_matrix = cupyx.scipy.sparse.coo_matrix(scipy_matrix)
    cupy_csr_matrix = cupy_coo_matrix.tocsr()
    cupy_csc_matrix = cupy_coo_matrix.tocsc()
    print("Sparse matrix-matrix")
    print(benchmark(matrix_matrix_coo, (cupy_coo_matrix, cupy_coo_matrix,), n_repeat=repeats))
    print(benchmark(matrix_matrix_csc, (cupy_csc_matrix, cupy_csc_matrix,), n_repeat=repeats))
    print(benchmark(matrix_matrix_csr, (cupy_csr_matrix, cupy_csr_matrix,), n_repeat=repeats))

    # vector-matrix dense 1d
    vector_dense_1d = cupy.zeros((nodes, ))
    vector_dense_1d[0] = 1
    print("Dense 1d vector-matrix")
    print(benchmark(vector_matrix_coo, (vector_dense_1d, cupy_coo_matrix,), n_repeat=repeats))
    print(benchmark(vector_matrix_csc, (vector_dense_1d, cupy_csc_matrix,), n_repeat=repeats))
    print(benchmark(vector_matrix_csr, (vector_dense_1d, cupy_csr_matrix,), n_repeat=repeats))

    # vector-matrix dense 2d
    vector_dense_2d = cupy.zeros((1, nodes))
    vector_dense_2d[0][0] = 1
    print("Dense 2d vector-matrix")
    print(benchmark(vector_matrix_coo, (vector_dense_2d, cupy_coo_matrix,), n_repeat=repeats))
    print(benchmark(vector_matrix_csc, (vector_dense_2d, cupy_csc_matrix,), n_repeat=repeats))
    print(benchmark(vector_matrix_csr, (vector_dense_2d, cupy_csr_matrix,), n_repeat=repeats))

    # vector_matrix sparse 2d
    vector_sparse_2d_coo = cupyx.scipy.sparse.coo_matrix(vector_dense_2d)
    vector_sparse_2d_csc = cupyx.scipy.sparse.csc_matrix(vector_dense_2d)
    vector_sparse_2d_csr = cupyx.scipy.sparse.csr_matrix(vector_dense_2d)
    print("Sparse 2d vector-matrix")
    print(benchmark(vector_matrix_coo, (vector_sparse_2d_coo, cupy_coo_matrix,), n_repeat=repeats))
    print(benchmark(vector_matrix_csc, (vector_sparse_2d_csc, cupy_csc_matrix,), n_repeat=repeats))
    print(benchmark(vector_matrix_csr, (vector_sparse_2d_csr, cupy_csr_matrix,), n_repeat=repeats))


if __name__ == "__main__":
    setup(90_000, 30_000)

