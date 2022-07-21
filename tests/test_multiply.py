import logging
import numpy as np

from csr import CSR
from csr.test_utils import mm_pairs, csr_slow

from pytest import approx
from hypothesis import given, settings, assume
from hypothesis import strategies as st

_log = logging.getLogger(__name__)


@csr_slow()
@given(st.data())
def test_multiply(kernel, data):
    A, B = data.draw(mm_pairs())
    assume(B.nnz < kernel.max_nnz)
    dA = A.to_scipy().toarray()
    dB = B.to_scipy().toarray()

    prod = A.multiply(B)
    assert isinstance(prod, CSR)
    _log.debug('got %r', prod)
    assert prod.nrows == A.nrows
    assert prod.ncols == B.ncols

    dprod = dA @ dB
    abnr, abnc = dprod.shape
    assert prod.nrows == abnr
    assert prod.ncols == abnc

    # assert prod.nnz == sp_prod.nnz
    if prod.nnz > 0:
        assert prod.values is not None
        assert np.all(prod.values != 0)
    nrows = A.nrows

    for i in range(nrows):
        r_scipy = dprod[i, :]
        r_ours = prod.row(i)

        assert len(r_ours) == len(r_scipy)
        assert r_ours == approx(r_scipy, rel=1.0e-5, abs=1.0e-10)


@csr_slow()
@given(st.data())
def test_multiply_transpose(kernel, data):
    A, B = data.draw(mm_pairs())
    assume(B.nnz < kernel.max_nnz)
    B = B.transpose()

    dA = A.to_scipy().toarray()
    dB = B.to_scipy().toarray()

    prod = A.multiply(B, transpose=True)
    assert isinstance(prod, CSR)
    _log.debug('got %r', prod)
    assert prod.nrows == A.nrows
    assert prod.ncols == B.nrows

    dprod = dA @ dB.T
    abnr, abnc = dprod.shape
    assert prod.nrows == abnr
    assert prod.ncols == abnc

    # assert prod.nnz == sp_prod.nnz
    if prod.nnz > 0:
        assert prod.values is not None
        assert np.all(prod.values != 0)
    nrows = A.nrows

    for i in range(nrows):
        r_scipy = dprod[i, :]
        r_ours = prod.row(i)

        assert len(r_ours) == len(r_scipy)
        assert r_ours == approx(r_scipy, rel=1.0e-5, abs=1.0e-10)
