import logging
import numpy as np
import scipy.sparse as sps

from csr import CSR
from csr.test_utils import csrs, csr_slow, mm_pairs

from pytest import mark, approx, raises
from hypothesis import given, assume, settings, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph

_log = logging.getLogger(__name__)


@settings(deadline=None)
@given(mm_pairs())
def test_multiply(pair):
    A, B = pair
    csra = CSR.from_scipy(A)
    csrb = CSR.from_scipy(B)

    prod = csra.multiply(csrb)
    assert isinstance(prod, CSR)
    _log.info('got %r', prod)
    assert prod.nrows == csra.nrows
    assert prod.ncols == csrb.ncols

    AB = A @ B
    abnr, abnc = AB.shape
    assert prod.nrows == abnr
    assert prod.ncols == abnc

    assert prod.nnz == AB.nnz
    nrows = csra.nrows

    for i in range(nrows):
        r_scipy = AB.getrow(i).toarray().ravel()
        r_ours = prod.row(i)

        assert len(r_ours) == len(r_scipy)
        assert r_ours == approx(r_scipy)
