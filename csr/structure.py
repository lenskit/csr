"""
Routines for working with matrix structure.
"""

import numpy as np
from numba import njit
from .csr import CSR
from . import _util
from .constructors import create, create_novalues

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


def subset_rows(csr, begin, end):
    "Take a subset of the rows of a CSR."
    st = csr.rowptrs[begin]
    ed = csr.rowptrs[end]
    rps = csr.rowptrs[begin:(end + 1)] - st

    cis = csr.colinds[st:ed]
    if csr.values is not None:
        vs = csr.values[st:ed]
    else:
        vs = None
    return CSR(end - begin, csr.ncols, ed - st, rps, cis, vs)


@njit(nogil=True)
def _pick_rows_nvs(csr: CSR, rows: np.ndarray):
    # how many values do we need
    # this is equivalent to np.sum(tag_mat.row_nnzs()[items]) (which would be fast in Python),
    # but uses less memory
    nr = len(rows)
    nnz = 0
    for i in rows:
        sp, ep = csr.row_extent(i)
        rl = ep - sp
        nnz += rl

    # allocate result arrays
    rowptrs = np.empty(nr + 1, dtype=np.int32)
    colinds = np.empty(nnz, dtype=np.int32)
    pos = 0

    # copy each item's rows to result array
    for ii, i in enumerate(rows):
        # get row start/end
        sp, ep = csr.row_extent(i)
        # how many tags for this item?
        itc = ep - sp
        end = pos + itc
        # copy values
        colinds[pos:end] = csr.colinds[sp:ep]
        # set offset
        rowptrs[ii] = pos
        # update position for storing results
        pos = end
    rowptrs[nr] = pos

    return CSR(nr, csr.ncols, nnz, rowptrs, colinds, None)


@njit(nogil=True)
def _pick_rows(csr: CSR, rows: np.ndarray):
    # how many values do we need
    # this is equivalent to np.sum(tag_mat.row_nnzs()[items]) (which would be fast in Python),
    # but uses less memory
    nr = len(rows)
    nnz = 0
    for i in rows:
        sp, ep = csr.row_extent(i)
        rl = ep - sp
        nnz += rl

    # allocate result arrays
    rowptrs = np.empty(nr + 1, dtype=np.int32)
    colinds = np.empty(nnz, dtype=np.int32)
    values = np.empty(nnz, dtype=csr.values.dtype)
    pos = 0

    # copy each item's rows to result array
    for ii, i in enumerate(rows):
        # get row start/end
        sp, ep = csr.row_extent(i)
        # how many tags for this item?
        itc = ep - sp
        end = pos + itc
        # copy values
        colinds[pos:end] = csr.colinds[sp:ep]
        values[pos:end] = csr.values[sp:ep]
        # set offset
        rowptrs[ii] = pos
        # update position for storing results
        pos = end
    rowptrs[nr] = pos

    return CSR(nr, csr.ncols, nnz, rowptrs, colinds, values)


@njit(nogil=True)
def sort_rows(csr):
    "Sort the rows of a CSR by increasing column index"
    for i in range(csr.nrows):
        sp, ep = csr.row_extent(i)
        # bubble-sort so it's super-fast on sorted arrays
        swapped = True
        while swapped:
            swapped = False
            for j in range(sp, ep - 1):
                if csr.colinds[j] > csr.colinds[j + 1]:
                    _util.maybe_swap(csr.colinds, j, j + 1)
                    _util.maybe_swap(csr.values, j, j + 1)
                    swapped = True


@njit(nogil=True)
def _transpose_values(csr):
    "Transpose a CSR with its values."
    brp = np.zeros(csr.ncols + 1, csr.rowptrs.dtype)
    bci = np.zeros(csr.nnz, np.int32)
    bvs = np.zeros(csr.nnz, np.float64)

    # count elements
    for i in range(csr.nrows):
        ars, are = csr.row_extent(i)
        for jj in range(ars, are):
            j = csr.colinds[jj]
            brp[j + 1] += 1

    # convert to pointers
    for j in range(csr.ncols):
        brp[j+1] = brp[j] + brp[j+1]

    # construct results
    for i in range(csr.nrows):
        ars, are = csr.row_extent(i)
        for jj in range(ars, are):
            j = csr.colinds[jj]
            bci[brp[j]] = i
            bvs[brp[j]] = csr.values[jj]
            brp[j] += 1

    # restore pointers
    for i in range(csr.ncols - 1, 0, -1):
        brp[i] = brp[i - 1]
    brp[0] = 0

    return create(csr.ncols, csr.nrows, csr.nnz, brp, bci, bvs)


@njit(nogil=True)
def _transpose_structure(csr):
    "Transpose a CSR, structure-only."
    brp = np.zeros(csr.ncols + 1, csr.rowptrs.dtype)
    bci = np.zeros(csr.nnz, np.int32)

    # count elements
    for i in range(csr.nrows):
        ars, are = csr.row_extent(i)
        for jj in range(ars, are):
            j = csr.colinds[jj]
            brp[j + 1] += 1

    # convert to pointers
    for j in range(csr.ncols):
        brp[j + 1] = brp[j] + brp[j+1]

    # construct results
    for i in range(csr.nrows):
        ars, are = csr.row_extent(i)
        for jj in range(ars, are):
            j = csr.colinds[jj]
            bci[brp[j]] = i
            brp[j] += 1

    # restore pointers
    for i in range(csr.ncols - 1, 0, -1):
        brp[i] = brp[i - 1]
    brp[0] = 0

    return create_novalues(csr.ncols, csr.nrows, csr.nnz, brp, bci)


def transpose(csr, include_values):
    if csr.values is None:
        include_values = False

    if include_values:
        return _transpose_values(csr)
    else:
        return _transpose_structure(csr)
