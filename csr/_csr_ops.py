"""
Backend implementations of Numba operations.
"""

import logging
import numpy as np
from numba import njit

from .array import swap

_log = logging.getLogger(__name__)


@njit(nogil=True)
def _center_rows(csr):
    means = np.zeros(csr.nrows)
    for i in range(csr.nrows):
        sp, ep = csr.row_extent(i)
        if sp == ep:
            continue  # empty row
        vs = csr.row_vs(i)
        m = np.mean(vs)
        means[i] = m
        csr.values[sp:ep] -= m

    return means


@njit(nogil=True)
def _unit_rows(csr):
    norms = np.zeros(csr.nrows)
    for i in range(csr.nrows):
        sp, ep = csr.row_extent(i)
        if sp == ep:
            continue  # empty row
        vs = csr.row_vs(i)
        m = np.linalg.norm(vs)
        norms[i] = m
        csr.values[sp:ep] /= m

    return norms


@njit(nogil=True)
def _csr_align(rowinds, nrows, rowptrs, align):
    rcts = np.zeros(nrows, dtype=rowptrs.dtype)
    for r in rowinds:
        rcts[r] += 1

    rowptrs[1:] = np.cumsum(rcts)
    rpos = rowptrs[:-1].copy()

    for i in range(len(rowinds)):
        row = rowinds[i]
        pos = rpos[row]
        align[pos] = i
        rpos[row] += 1


@njit(nogil=True)
def _csr_transpose(shape, arp, aci, avs):
    nrows, ncols = shape
    brp = np.zeros(ncols + 1, arp.dtype)
    bci = np.zeros(len(aci), np.int32)
    if avs is not None:
        bvs = np.zeros(len(aci), np.float64)
    else:
        bvs = None

    # count elements
    for i in range(nrows):
        ars = arp[i]
        are = arp[i+1]
        for jj in range(ars, are):
            j = aci[jj]
            brp[j+1] += 1

    # convert to pointers
    for j in range(ncols):
        brp[j+1] = brp[j] + brp[j+1]

    # construct results
    for i in range(nrows):
        ars = arp[i]
        are = arp[i+1]
        for jj in range(ars, are):
            j = aci[jj]
            bci[brp[j]] = i
            if bvs is not None:
                bvs[brp[j]] = avs[jj]
            brp[j] += 1

    # restore pointers
    for i in range(ncols-1, 0, -1):
        brp[i] = brp[i-1]
    brp[0] = 0

    return brp, bci, bvs


@njit(nogil=True)
def _csr_align_inplace(shape, rows, cols, vals):
    """
    Align COO data in-place for a CSR matrix.

    Args:
        shape: the matrix shape
        rows: the matrix row indices (not modified)
        cols: the matrix column indices (**modified**)
        vals: the matrix values (**modified**)

    Returns:
        the CSR row pointers
    """
    nrows, ncols = shape
    nnz = len(rows)

    rps = np.zeros(nrows + 1, np.int64)

    for i in range(nnz):
        rps[rows[i] + 1] += 1
    for i in range(nrows):
        rps[i+1] += rps[i]

    rci = rps[:nrows].copy()

    pos = 0
    row = 0
    rend = rps[1]

    # skip to first nonempty row
    while row < nrows and rend == 0:
        row += 1
        rend = rps[row + 1]

    while pos < nnz:
        r = rows[pos]
        # swap until we have something in place
        while r != row:
            tgt = rci[r]
            # swap with the target position
            swap(cols, pos, tgt)
            if vals is not None:
                swap(vals, pos, tgt)

            # update the target start pointer
            rci[r] += 1

            # update the loop check
            r = rows[tgt]

        # now the current entry in the arrays is good
        # we need to advance to the next entry
        pos += 1
        rci[row] += 1

        # skip finished rows
        while pos == rend and pos < nnz:
            row += 1
            pos = rci[row]
            rend = rps[row+1]

    return rps
