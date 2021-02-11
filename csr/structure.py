"""
Routines for working with matrix structure.
"""

import numpy as np
from numba import njit


@njit(nogil=True)
def _from_coo_structure(nrows, rows, cols):
    "Transform COO structure into CSR structure"
    nnz = len(rows)
    counts = np.zeros(nrows, dtype=np.int64)
    for r in rows:
        counts[r] += 1

    rowptrs = np.zeros(nrows + 1, dtype=np.int64)
    for i in range(nrows):
        rowptrs[i + 1] = rowptrs[i] + counts[i]

    rpos = rowptrs.copy()
    out_cols = np.empty_like(cols)

    for i in range(nnz):
        row = rows[i]
        pos = rpos[row]
        out_cols[pos] = cols[i]
        rpos[row] += 1

    return rowptrs, out_cols


@njit(nogil=True)
def _from_coo_values(nrows, rows, cols, values):
    "Transform COO w/ values into CSR"
    nnz = len(rows)
    counts = np.zeros(nrows, dtype=np.int64)
    for r in rows:
        counts[r] += 1

    rowptrs = np.zeros(nrows + 1, dtype=np.int64)
    for i in range(nrows):
        rowptrs[i + 1] = rowptrs[i] + counts[i]

    rpos = rowptrs.copy()
    out_cols = np.empty_like(cols)
    out_vals = np.empty_like(values)

    for i in range(nnz):
        row = rows[i]
        pos = rpos[row]
        out_cols[pos] = cols[i]
        out_vals[pos] = values[i]
        rpos[row] += 1

    return rowptrs, out_cols, out_vals


def from_coo(nrows, rows, cols, values=None):
    if values is None:
        rps, cols = _from_coo_structure(nrows, rows, cols)
        vals = None
    else:
        rps, cols, vals = _from_coo_values(nrows, rows, cols, values)
    return rps, cols, vals
