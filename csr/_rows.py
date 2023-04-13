"""
Implementations of CSR row access functions.
"""

import numpy as np
from numba import njit


def extent(csr, row):
    "Get the extent of a row in the matrix storage."
    sp = csr.rowptrs[row]
    ep = csr.rowptrs[row + 1]
    return sp, ep


@njit
def _fill_vals(csr, row, array):
    sp, ep = csr.row_extent(row)
    cols = csr.colinds[sp:ep]
    array[cols] = csr.values[sp:ep]


@njit
def _fill_ones(csr, row, array):
    sp, ep = csr.row_extent(row)
    cols = csr.colinds[sp:ep]
    array[cols] = 1


@njit
def _row_array(csr, row, fill, dtype):
    v = np.zeros(csr.ncols, dtype=dtype)
    if csr.nnz == 0:
        return v

    fill(csr, row, v)
    return v


@njit
def _mr_matrix(csr, rows, fill, dtype):
    v = np.zeros(rows.shape + (csr.ncols,), dtype=dtype)
    if csr.nnz == 0:
        return v

    for i, row in enumerate(rows):
        fill(csr, row, v[i, :])

    return v


def _row_array_vals(csr, row):
    return _row_array(csr, row, _fill_vals, csr.values.dtype)


def _row_array_ones(csr, row):
    return _row_array(csr, row, _fill_ones, np.float32)


def _row_array_mask(csr, row):
    return _row_array(csr, row, _fill_ones, np.bool_)


def _mr_matrix_vals(csr, row):
    return _mr_matrix(csr, row, _fill_vals, csr.values.dtype)


def _mr_matrix_ones(csr, row):
    return _mr_matrix(csr, row, _fill_ones, np.float32)


def _mr_matrix_mask(csr, row):
    return _mr_matrix(csr, row, _fill_ones, np.bool_)


def row_array(csr, row):
    "Get a row of the CSR as an array"
    if row.shape == ():
        if csr.values is not None:
            return _row_array_vals(csr, row)
        else:
            return _row_array_ones(csr, row)
    else:
        if csr.values is not None:
            return _mr_matrix_vals(csr, row)
        else:
            return _mr_matrix_ones(csr, row)


def row_mask(csr, row):
    "Get a row of the CSR as an array"
    if row.shape == ():
        return _row_array_mask(csr, row)
    else:
        return _mr_matrix_mask(csr, row)


def cs(csr, row):
    "Get the column indices for a row."
    sp, ep = csr.row_extent(row)
    return csr.colinds[sp:ep]


def _vs_vals(csr, row):
    sp, ep = csr.row_extent(row)
    return csr.values[sp:ep]


def _vs_ones(csr, row):
    sp, ep = csr.row_extent(row)
    return np.full(ep - sp, 1.0)


def vs(csr, row):
    "Get the nonzero values for a row."
    if csr.values is not None:
        return _vs_vals(csr, row)
    else:
        return _vs_ones(csr, row)


def all_indices(csr):
    "Get the row indices for the nonzero values in a matrix."
    ris = np.zeros(csr.nnz, np.intc)
    for i in range(csr.nrows):
        sp, ep = csr.row_extent(i)
        ris[sp:ep] = i
    return ris
