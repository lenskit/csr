import logging

from csr import CSR
from csr.test_utils import mm_pairs

from pytest import approx
from hypothesis import given, settings

_log = logging.getLogger(__name__)


@settings(deadline=None)
@given(mm_pairs())
def test_multiply(kernel, pair):
    A, B = pair
    csra = CSR.from_scipy(A)
    csrb = CSR.from_scipy(B)

    prod = csra.multiply(csrb)
    assert isinstance(prod, CSR)
    _log.info('got %r', prod)
    _log.info('inner: %s', prod.R)
    assert prod.nrows == csra.nrows
    assert prod.ncols == csrb.ncols

    AB = A @ B
    abnr, abnc = AB.shape
    assert prod.nrows == abnr
    assert prod.ncols == abnc

    assert prod.nnz == AB.nnz
    if prod.nnz > 0:
        assert prod.values is not None
    nrows = csra.nrows

    for i in range(nrows):
        r_scipy = AB.getrow(i).toarray().ravel()
        r_ours = prod.row(i)

        assert len(r_ours) == len(r_scipy)
        assert r_ours == approx(r_scipy)


@settings(deadline=None)
@given(mm_pairs())
def test_multiply_transpose(kernel, pair):
    A, B = pair
    csra = CSR.from_scipy(A)
    csrb = CSR.from_scipy(B.T)

    prod = csra.multiply(csrb, transpose=True)
    assert isinstance(prod, CSR)
    _log.info('got %r', prod)
    _log.info('inner: %s', prod.R)
    assert prod.nrows == csra.nrows
    assert prod.ncols == csrb.nrows

    AB = A @ B
    abnr, abnc = AB.shape
    assert prod.nrows == abnr
    assert prod.ncols == abnc

    assert prod.nnz == AB.nnz
    if prod.nnz > 0:
        assert prod.values is not None
    nrows = csra.nrows

    for i in range(nrows):
        r_scipy = AB.getrow(i).toarray().ravel()
        r_ours = prod.row(i)

        assert len(r_ours) == len(r_scipy)
        assert r_ours == approx(r_scipy)
