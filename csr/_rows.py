"""
Implementations of CSR row access functions.
"""

import numpy as np


def extent(csr, row):
    "Get the extent of a row in the matrix storage."
    sp = csr.rowptrs[row]
    ep = csr.rowptrs[row + 1]
    return sp, ep


def _array_vals(csr, row):
    "Get a row as a dense vector."
    v = np.zeros(csr.ncols)
    if csr.nnz == 0:
        return v

    sp, ep = csr.row_extent(row)
    cols = csr.colinds[sp:ep]
    v[cols] = csr.values[sp:ep]

    return v


def _array_ones(csr, row):
    v = np.zeros(csr.ncols)
    if csr.nnz == 0:
        return v

    sp, ep = csr.row_extent(row)
    cols = csr.colinds[sp:ep]
    v[cols] = 1

    return v


def array(csr, row):
    if csr.values is not None:
        return _array_vals(csr, row)
    else:
        return _array_ones(csr, row)


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
