import sys
from typing import Optional
import dataclasses
from numba.core.types.functions import _ResolutionFailures
import numpy as np
from numba import njit, config
from numba.extending import overload
from numba.core.types import StructRef, intc, float64
from numba.experimental import structref

from csr import CSR
from ._api import *  # noqa: F403
import csr.kernels.mkl as _hpkg
from csr.constructors import create_empty

__all__ = [
    'mkl_h',
    'to_handle',
    'from_handle',
    'release_handle'
]


@structref.register
class mkl_h_type(StructRef):
    "Internal Numba type for MKL handles"
    pass


if config.DISABLE_JIT:
    @dataclasses.dataclass
    class mkl_h:
        H: int
        nrows: int
        ncols: int
        csr_ref: Optional[np.ndarray]
else:
    class mkl_h(structref.StructRefProxy):
        """
        Type for MKL handles.  Opaque, do not use directly.
        """

    structref.define_proxy(mkl_h, mkl_h_type, ['H', 'nrows', 'ncols', 'csr_ref'])


def _make_handle_impl(csr):
    "Make a handle from a known-constructable CSR"
    _sp = ffi.from_buffer(csr.rowptrs)
    _cols = ffi.from_buffer(csr.colinds)
    vs = csr.values
    assert vs.size == csr.nnz
    _vals = ffi.from_buffer(vs)
    h = lk_mkl_spcreate(csr.nrows, csr.ncols, _sp, _cols, _vals)
    lk_mkl_spopt(h)
    return mkl_h(h, csr.nrows, csr.ncols, csr)


_make_handle = njit(_make_handle_impl)


def to_handle(csr: CSR) -> mkl_h:
    if csr.nnz > _hpkg.max_nnz:
        raise ValueError('CSR size {} exceeds max nnz {}'.format(csr.nnz, _hpkg.max_nnz))

    if csr.nnz == 0:
        # empty matrices don't really work
        return mkl_h(0, csr.nrows, csr.ncols, None)

    norm = csr._normalize(np.float64, np.intc)
    return _make_handle(norm)


@overload(to_handle)
def to_handle_jit(csr):
    if csr.ptr_type.dtype != intc:
        raise TypeError('MKL requires intc row pointers')

    if csr.has_values:
        vt = csr.val_type.dtype
    else:
        vt = None

    def mkh(csr):
        vs = csr._required_values().astype(np.float64)
        csr2 = CSR(csr.nrows, csr.ncols, csr.nnz, csr.rowptrs, csr.colinds, vs)

        if csr.nnz == 0:
            return mkl_h(0, csr.nrows, csr.ncols, csr2)

        return _make_handle(csr2)

    return mkh


@njit
def from_handle(h: mkl_h) -> CSR:
    if not h.H:
        return create_empty(h.nrows, h.ncols)

    rvp = lk_mkl_spexport_p(h.H)
    if rvp is None:
        return None

    nrows = lk_mkl_spe_nrows(rvp)
    ncols = lk_mkl_spe_ncols(rvp)

    sp = lk_mkl_spe_row_sp(rvp)
    ep = lk_mkl_spe_row_ep(rvp)
    cis = lk_mkl_spe_colinds(rvp)
    vs = lk_mkl_spe_values(rvp)

    rowptrs = np.zeros(nrows + 1, dtype=np.intc)
    nnz = 0
    for i in range(nrows):
        nnz += ep[i] - sp[i]
        rowptrs[i + 1] = nnz
    assert nnz == ep[nrows - 1]

    colinds = np.zeros(nnz, dtype=np.intc)
    values = np.zeros(nnz)

    for i in range(nrows):
        rs = rowptrs[i]
        re = rowptrs[i + 1]
        ss = sp[i]
        for j in range(re - rs):
            colinds[rs + j] = cis[ss + j]
            values[rs + j] = vs[ss + j]

    lk_mkl_spe_free(rvp)

    return CSR(nrows, ncols, nnz, rowptrs, colinds, values)


@njit
def order_columns(h):
    """
    Sort matrix rows in increasing column order.
    """
    if h.H:
        lk_mkl_sporder(h.H)


@njit
def release_handle(h: mkl_h):
    if h.H:
        lk_mkl_spfree(h.H)
    h.H = 0
