from typing import Any, Iterable, TypedDict, TypeVar

from numpy import dtype, ndarray
from scipy.sparse import (
    coo_array,
    csc_array,
    csc_matrix,
    csr_array,
    csr_matrix,
)


class NumpyArrayDict(TypedDict):
    shape: Iterable[int]
    size: int
    ndim: int
    data_nbytes: int
    data_dtype: dtype[Any]
    data_itemsize: int
    data_size: int


class ScipySparseArrayDict(NumpyArrayDict):
    nnz: int
    format: str


class ScipyCSArrayDict(ScipySparseArrayDict):
    indices_nbytes: int
    indices_dtype: dtype[Any]
    indices_itemsize: int
    indices_size: int
    indptr_nbytes: int
    indptr_dtype: dtype[Any]
    indptr_itemsize: int
    indptr_size: int
    has_sorted_indices: bool


class ScipyCOOArrayDict(ScipySparseArrayDict):
    row_nbytes: int
    row_dtype: dtype[Any]
    row_itemsize: int
    row_size: int
    col_nbytes: int
    col_dtype: dtype[Any]
    col_itemsize: int
    col_size: int


def numpy_array_info(array: ndarray) -> NumpyArrayDict:
    return {
        "shape": array.shape,
        "size": array.size,
        "ndim": array.ndim,
        "data_nbytes": array.data.nbytes,
        "data_dtype": array.data.dtype,
        "data_itemsize": array.data.itemsize,
        "data_size": array.data.size,
    }


_CSLIKE = TypeVar("_CSLIKE", csc_array, csc_matrix, csr_array, csr_matrix)


def scipy_cs_array_info(array: _CSLIKE) -> ScipyCSArrayDict:
    """Return information of a CSC or CSR array/matrix as a dictionary.

    Function which maps various properties of a Compressed Sparse Column
    (CSC) or Compressed Sparse Row (CSR) array/matrix to a dictionary.
    Useful when writing this data to a CSV-file.

    :param array: CSC or CSR array/matrix to map to a dictionary.
    :type array: _CSLIKE
    :return: Dictionary with information about the array/matrix.
    :rtype ScipyCSArrayDict
    """
    return {
        "shape": array.shape,
        "size": array.size,
        "ndim": array.ndim,
        "data_nbytes": array.data.nbytes,
        "data_dtype": array.data.dtype,
        "data_itemsize": array.data.itemsize,
        "data_size": array.data.size,
        "nnz": array.nnz,
        "format": array.format,
        "indices_nbytes": array.indices.nbytes,
        "indices_dtype": array.indices.dtype,
        "indices_itemsize": array.indices.itemsize,
        "indices_size": array.indices.size,
        "indptr_nbytes": array.indptr.nbytes,
        "indptr_dtype": array.indptr.dtype,
        "indptr_itemsize": array.indptr.itemsize,
        "indptr_size": array.indptr.size,
        "has_sorted_indices": array.has_sorted_indices,
    }


def scipy_coo_array_info(array: coo_array) -> ScipyCOOArrayDict:
    return {
        "shape": array.shape,
        "size": array.size,
        "ndim": array.ndim,
        "data_nbytes": array.data.nbytes,
        "data_dtype": array.data.dtype,
        "data_itemsize": array.data.itemsize,
        "data_size": array.data.size,
        "nnz": array.nnz,
        "format": array.format,
        "row_nbytes": array.row.nbytes,
        "row_dtype": array.row.dtype,
        "row_itemsize": array.row.itemsize,
        "row_size": array.row.size,
        "col_nbytes": array.col.nbytes,
        "col_dtype": array.col.dtype,
        "col_itemsize": array.col.itemsize,
        "col_size": array.col.size,
    }
