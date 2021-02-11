import numpy as np
from numba import njit

from ._api import *  # noqa: F403
from .handle import mkl_h

__all__ = [
    'mult_ab',
    'mult_abt'
]


@njit(nogil=True)
def mult_ab(a_h, b_h):
    if a_h.H and b_h.H:
        h = lk_mkl_spmab(a_h.H, b_h.H)
    else:
        h = 0
    return mkl_h(h, a_h.nrows, b_h.ncols, None)


@njit(nogil=True)
def mult_abt(a_h, b_h):
    if a_h.H and b_h.H:
        h = lk_mkl_spmabt(a_h.H, b_h.H)
    else:
        h = 0
    return mkl_h(h, a_h.nrows, b_h.nrows, None)


@njit(nogil=True)
def mult_vec(a_h, x):
    y = np.zeros(a_h.nrows, dtype=np.float64)

    if a_h.H:
        _x = ffi.from_buffer(x)
        _y = ffi.from_buffer(y)

        lk_mkl_spmv(1.0, a_h.H, _x, 0.0, _y)

    return y
