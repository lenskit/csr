"""
Various MKL-specific tests.
"""

import pytest
from numba import njit
import numpy as np

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
