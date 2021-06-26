"""
Various MKL-specific tests.
"""

import pytest
from numba import njit
import numpy as np

from csr import CSR
from csr.test_utils import csrs

from hypothesis import given

try:
    from csr.kernels import mkl
except ImportError:
    pytestmark = pytest.skip("MKL is not available")


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
        print('allocating indexes')
        colinds = np.empty(nnz, dtype=np.intc)
        print('allocating values')
        values = np.random.randn(nnz)
    except MemoryError:
        pytest.skip('insufficient memory')

    print('randomizing colinds')
    for i in range(nrows):
        s = i * dense
        e = s + dense
        colinds[s:e] = np.random.choice(ncols, dense, replace=False)

    csr = CSR(nrows, ncols, nnz, rowptrs, colinds, values)

    v = np.random.nrand(ncols)

    res = csr.mult_vec(v)

    assert res.shape == (nrows,)
    assert np.all(~np.isnan(res))
