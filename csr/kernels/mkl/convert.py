import numpy as np
from numba import njit

from csr import _CSR
from . import _mkl_ops

ffi = _mkl_ops.ffi
lib = _mkl_ops.lib

__all__ = [
    'to_handle',
    'from_handle',
    'release_handle'
]


@njit
def to_handle(csr):
    """
    Convert a native CSR to a handle.  The caller must arrange for the CSR last at
    least as long as the handle.  The handle must be explicitly released.
    """
    _sp = ffi.from_buffer(csr.rowptrs)
    _cols = ffi.from_buffer(csr.colinds)
    _vals = ffi.from_buffer(csr.values)
    return lib.lk_mkl_spcreate(csr.nrows, csr.ncols, _sp, _cols, _vals)


@njit
def from_handle(h):
    """
    Convert a handle to a CSR.  The handle may be released after this is called.
    """

    rvp = lib.mkl_mkl_spexport_p(h)
    if rvp is None:
        return None

    nrows = lib.mkl_mkl_spe_nrows(rvp)
    ncols = lib.mkl_mkl_spe_ncols(rvp)

    sp = lib.mkl_mkl_spe_row_sp(rvp)
    ep = lib.mkl_mkl_spe_row_ep(rvp)
    cis = lib.mkl_mkl_spe_colinds(rvp)
    vs = lib.mkl_mkl_spe_values(rvp)

    rowptrs = np.zeros(nrows + 1, dtype=np.intc)
    nnz = 0
    for i in range(nrows):
        nnz += ep[i] - sp[i]
        rowptrs[i+1] = nnz

    colinds = np.zeros(nnz, dtype=np.intc)
    values = np.zeros(nnz)

    for i in range(nrows):
        rs = rowptrs[i]
        re = rowptrs[i+1]
        ss = sp[i]
        for j in range(re - rs):
            colinds[rs + j] = cis[ss + j]
            values[rs + j] = vs[ss + j]

    lib.mkl_mkl_spe_free(rvp)

    return _CSR(nrows, ncols, nnz, rowptrs, colinds, values)


@njit
def release_handle(h):
    """
    Release a handle.
    """
    lib.lk_mkl_spfree(h)
