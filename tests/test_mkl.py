"""
Various MKL-specific tests.
"""

import logging
from contextlib import contextmanager
from numba import njit, prange
import numpy as np

from csr import CSR
from csr.test_utils import csrs, has_memory
from csr.kernels import use_kernel

from pytest import skip, mark
from hypothesis import given

import test_multiply as tmm
import test_mult_vec as tmv


try:
    from csr.kernels import mkl
except ImportError:
    pytestmark = mark.skip("MKL is not available")

_log = logging.getLogger(__name__)


@contextmanager
def mkl_lim(lim=1000):
    "Limit MKL to a capacity of X"
    save = mkl.max_nnz
    try:
        mkl.max_nnz = lim
        yield lim
    finally:
        mkl.max_nnz = save
        pass


@njit
def make_handle(csr):
    return mkl.to_handle(csr)


@njit
def unhandle(h):
    return mkl.from_handle(h)


@given(csrs())
def test_csr_handle(csr):
    h = make_handle(csr)
    csr2 = unhandle(h)

    assert csr2.nrows == csr.nrows
    assert csr2.ncols == csr.ncols
    assert csr2.nnz == csr.nnz
    assert np.all(csr2.rowptrs == csr.rowptrs)
    assert np.all(csr2.colinds == csr.colinds)
    if csr.values is not None:
        assert np.all(csr2.values == csr.values)
    else:
        assert np.all(csr2.values == 1.0)


@njit(parallel=True)
def fill_rows(values, colinds, nrows, ncols, dense):
    for i in prange(nrows):
        s = i * dense
        e = s + dense
        values[s:e] = np.random.randn(e - s)
        colinds[s:e] = np.random.choice(ncols, dense, replace=False)


def test_mult_vec_lim():
    "Test matrix-vector multiply with limited kernel capacity"
    with mkl_lim(), use_kernel('mkl'):
        tmv.test_mult_vec(mkl)


def test_multiply_lim():
    "Test matrix-matrix multiply with limited kernel capacity"
    with mkl_lim(), use_kernel('mkl'):
        tmm.test_multiply(mkl)


def test_multiply_transpose_lim():
    "Test matrix-matrix transpose multiply with limited kernel capacity"
    with mkl_lim(), use_kernel('mkl'):
        tmm.test_multiply_transpose(mkl)


@mark.skipif(not has_memory(32), reason='insufficient memory')
def test_large_mult_vec():
    # 10M * 500 = 2.5B >= INT_MAX
    nrows = 10000000
    ncols = 500
    dense = 250
    nnz = nrows * dense

    rowptrs = np.arange(0, nnz + 1, dense, dtype=np.int64)

    assert len(rowptrs) == nrows + 1
    assert rowptrs[-1] == nnz

    try:
        _log.info('allocating indexes')
        colinds = np.empty(nnz, dtype=np.intc)
        _log.info('allocating values')
        values = np.zeros(nnz)
    except MemoryError:
        skip('insufficient memory')

    _log.info('randomizing array contents')
    fill_rows(values, colinds, nrows, ncols, dense)

    csr = CSR(nrows, ncols, nnz, rowptrs, colinds, values)

    v = np.random.randn(ncols)

    res = csr.mult_vec(v)

    assert res.shape == (nrows,)
    assert np.all(~np.isnan(res))
