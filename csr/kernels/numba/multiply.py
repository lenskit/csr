"""
Matrix multiplication using the SMMP algorithm [SMMP]_.

.. [SMMP] Bank, R. E., & Douglas, C. C. (1993). Sparse matrix multiplication package (SMMP).
    _Advances in Computational Mathematics_, 1(1), 127–137. <http://dx.doi.org/10.1007/BF02070824>
"""

import numpy as np
from numba import njit
from csr import CSR


@njit(nogil=True)
def mult_ab(a_h, b_h):
    """
    Multiply matrices A and B.

    Args:
        a_h: the handle of matrix A
        b_h: the handle of matrix B

    Returns:
        the handle of the product; it must be released when no longer needed.
    """

    assert a_h.ncols == b_h.nrows

    c_rp = np.zeros(a_h.nrows + 1, np.intc)

    # step 1: symbolic multiplication
    c_ci = _sym_mm(a_h, b_h, c_rp)
    c_nnz = c_rp[a_h.nrows]

    # step 2: numeric multiplication
    c_vs = _num_mm(a_h, b_h, c_rp, c_ci)

    # build the result
    return CSR(a_h.nrows, b_h.ncols, c_nnz, c_rp, c_ci, c_vs)


@njit(nogil=True)
def mult_abt(a_h, b_h):
    """
    Multiply matrices A and B^T.

    Args:
        a_h: the handle of matrix A
        b_h: the handle of matrix B

    Returns:
        the handle of the product; it must be released when no longer needed.
    """
    assert a_h.ncols == b_h.ncols

    # transpose B
    bt_h = b_h.transpose()
    return mult_ab(a_h, bt_h)


@njit
def _sym_mm(a_h, b_h, c_rp):
    wlen = max(a_h.nrows, a_h.ncols, b_h.ncols)
    index = np.full(wlen, -1, np.intc)
    c_len = max(a_h.nnz, b_h.nnz)
    c_ci = np.zeros(c_len, np.intc)
    c_pos = 0

    for i in range(a_h.nrows):
        istart = wlen
        length = 0

        # Pass 1: count and link columns in row
        a_rs, a_re = a_h.row_extent(i)
        for jj in range(a_rs, a_re):
            j = a_h.colinds[jj]
            b_rs, b_re = b_h.row_extent(j)
            for kk in range(b_rs, b_re):
                k = b_h.colinds[kk]
                if index[k] < 0:
                    index[k] = istart
                    istart = k
                    length += 1

        # Make sure we have enough length
        while c_pos + length > c_len:
            ocl = c_len
            c_len = c_len + c_len // 2
            ci2 = np.empty(c_len, np.intc)
            ci2[:ocl] = c_ci
            c_ci = ci2

        # Pass 2: fill in the rows
        c_rp[i + 1] = c_rp[i] + length
        for j in range(c_rp[i], c_rp[i + 1]):
            c_ci[j] = istart
            istart = index[istart]
            index[c_ci[j]] = -1
        c_pos += length

    return c_ci[:c_pos]


@njit
def _num_mm(a_h, b_h, c_rp, c_ci):
    # set up work array
    wlen = max(a_h.nrows, a_h.ncols, b_h.ncols)
    work = np.zeros(wlen)
    # initalize result array
    c_vs = np.zeros(len(c_ci))

    for i in range(a_h.nrows):
        a_rs, a_re = a_h.row_extent(i)
        for jj in range(a_rs, a_re):
            j = a_h.colinds[jj]
            av = a_h.values[jj]

            b_rs, b_re = b_h.row_extent(j)
            for kk in range(b_rs, b_re):
                k = b_h.colinds[kk]
                work[k] += av * b_h.values[kk]

        c_rs = c_rp[i]
        c_re = c_rp[i + 1]
        for jj in range(c_rs, c_re):
            j = c_ci[jj]
            c_vs[jj] = work[j]
            work[j] = 0

    return c_vs
