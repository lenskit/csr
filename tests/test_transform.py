import numpy as np

from csr import CSR
from csr.test_utils import csrs, csr_slow, sparse_matrices

from pytest import mark, approx, raises
from hypothesis import given, assume, settings, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph


@csr_slow()
@given(sparse_matrices())
def test_mean_center(spm):
    # assume(spm.nnz >= 10)
    csr = CSR.from_scipy(spm)

    m2 = csr.normalize_rows('center')
    assert len(m2) == csr.nrows
    rnnz = csr.row_nnzs()

    for i in range(csr.nrows):
        vs = csr.row_vs(i)
        spr = spm[i, :].toarray()
        if rnnz[i] > 0:
            assert m2[i] == approx(np.sum(spr) / rnnz[i])
            assert np.mean(vs) == approx(0.0)
            assert vs + m2[i] == approx(spr[0, csr.row_cs(i)])


@csr_slow()
@given(sparse_matrices())
def test_unit_norm(spm):
    # assume(spm.nnz >= 10)
    csr = CSR.from_scipy(spm)

    m2 = csr.normalize_rows('unit')
    assert len(m2) == csr.nrows

    for i in range(csr.nrows):
        vs = csr.row_vs(i)
        if len(vs) > 0:
            assert np.linalg.norm(vs) == approx(1.0)
            assert vs * m2[i] == approx(spm.getrow(i).toarray()[0, csr.row_cs(i)])


@csr_slow()
@given(sparse_matrices())
def test_filter(mat):
    assume(mat.nnz > 0)
    assume(not np.all(mat.data <= 0))  # we have to have at least one to retain
    csr = CSR.from_scipy(mat)
    csrf = csr.filter_nnzs(csr.values > 0)
    assert all(csrf.values > 0)
    assert csrf.nnz <= csr.nnz

    for i in range(csr.nrows):
        spo, epo = csr.row_extent(i)
        spf, epf = csrf.row_extent(i)
        assert epf - spf <= epo - spo

    d1 = csr.to_scipy().toarray()
    df = csrf.to_scipy().toarray()
    d1[d1 < 0] = 0
    assert df == approx(d1)
