import logging

from csr import CSR
from csr.test_utils import mm_pairs

from pytest import approx
from hypothesis import given, settings
from hypothesis import strategies as st

_log = logging.getLogger(__name__)


@settings(deadline=None)
@given(st.data())
def test_multiply(kernel, data):
    A, B = data.draw(mm_pairs(max_nnz=kernel.max_nnz))
    spA = A.to_scipy()
    spB = B.to_scipy()

    prod = A.multiply(B)
    assert isinstance(prod, CSR)
    _log.info('got %r', prod)
    assert prod.nrows == A.nrows
    assert prod.ncols == B.ncols

    sp_prod = spA @ spB
    abnr, abnc = sp_prod.shape
    assert prod.nrows == abnr
    assert prod.ncols == abnc

    assert prod.nnz == sp_prod.nnz
    if prod.nnz > 0:
        assert prod.values is not None
    nrows = A.nrows

    for i in range(nrows):
        r_scipy = sp_prod.getrow(i).toarray().ravel()
        r_ours = prod.row(i)

        assert len(r_ours) == len(r_scipy)
        assert r_ours == approx(r_scipy)


@settings(deadline=None)
@given(st.data())
def test_multiply_transpose(kernel, data):
    A, B = data.draw(mm_pairs(max_nnz=kernel.max_nnz))
    B = B.transpose()

    spA = A.to_scipy()
    spB = B.to_scipy()

    prod = A.multiply(B, transpose=True)
    assert isinstance(prod, CSR)
    _log.info('got %r', prod)
    assert prod.nrows == A.nrows
    assert prod.ncols == B.nrows

    sp_prod = spA @ spB.T
    abnr, abnc = sp_prod.shape
    assert prod.nrows == abnr
    assert prod.ncols == abnc

    assert prod.nnz == sp_prod.nnz
    if prod.nnz > 0:
        assert prod.values is not None
    nrows = A.nrows

    for i in range(nrows):
        r_scipy = sp_prod.getrow(i).toarray().ravel()
        r_ours = prod.row(i)

        assert len(r_ours) == len(r_scipy)
        assert r_ours == approx(r_scipy)
