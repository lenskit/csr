import logging
import numpy as np
import scipy.sparse as sps

from csr import CSR
from csr.test_utils import csrs, csr_slow, sparse_matrices

from pytest import mark, approx, raises
from hypothesis import given, assume, settings, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph

_log = logging.getLogger(__name__)


@csr_slow()
@given(st.data())
def test_subset_rows(data):
    nrows = data.draw(st.integers(5, 100))
    ncols = data.draw(st.integers(1, 100))
    dens = data.draw(st.floats(0, 1))
    beg = data.draw(st.integers(0, nrows - 1))
    end = data.draw(st.integers(beg, nrows - 1))

    spm = sps.random(nrows, ncols, dens, format='csr')
    csr = CSR.from_scipy(spm)

    m2 = csr.subset_rows(beg, end)
    assert m2.nrows == end - beg

    for i in range(m2.nrows):
        assert all(m2.row_cs(i) == csr.row_cs(beg+i))
        assert all(m2.row_vs(i) == csr.row_vs(beg+i))


@csr_slow()
@given(st.data())
def test_pick_rows(data):
    include = data.draw(st.booleans())
    csr = data.draw(csrs())
    nrows = csr.nrows

    to_pick = st.integers(0, nrows * 10)
    r_range = st.integers(0, nrows - 1)
    rows = data.draw(nph.arrays(np.int32, to_pick, elements=r_range))

    _log.debug('picking %d rows from %d-row matrix', len(rows), nrows)

    sub = csr.pick_rows(rows, include_values=include)
    assert sub.nrows == len(rows)
    assert len(sub.rowptrs) == sub.nrows + 1
    if include and csr.values is not None:
        assert sub.values is not None
    else:
        assert sub.values is None

    for i, r in enumerate(rows):
        osp, oep = csr.row_extent(r)
        sp, ep = sub.row_extent(i)
        assert oep - osp == ep - sp

        assert all(csr.row_cs(r) == sub.row_cs(i))
        if sub.values is not None:
            assert all(csr.row_vs(r) == sub.row_vs(i))


@given(csrs())
def test_sort_rows(csr):
    tv = np.ones(csr.ncols)
    x1 = csr.mult_vec(tv)
    csr.sort_rows()
    assert all(all(np.diff(csr.row_cs(i)) > 0) for i in range(csr.nrows))
    x2 = csr.mult_vec(tv)
    assert x2 == approx(x1)


@given(csrs())
def test_kernel_sort_rows(kernel, csr):
    tv = np.ones(csr.ncols)
    x1 = csr.mult_vec(tv)
    h = kernel.to_handle(csr.R)
    kernel.order_columns(h)
    c2 = kernel.from_handle(h)
    kernel.release_handle(h)
    assert all(all(np.diff(c2.row_cs(i)) > 0) for i in range(csr.nrows))
    x2 = c2.mult_vec(tv)
    assert x2 == approx(x1)


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


# @csr_slow()
@given(sparse_matrices())
def test_shard(mat):
    SHARD_SIZE = 1000
    csr = CSR.from_scipy(mat)

    shards = csr._shard_rows(SHARD_SIZE)
    # we have the whole matrix
    assert sum(s.nnz for s in shards) == csr.nnz
    # everything is in spec
    assert all(s.nnz <= SHARD_SIZE for s in shards)

    # all row counts match
    assert np.all(np.concatenate([s.row_nnzs() for s in shards]) == csr.row_nnzs())
    # all column indices match
    assert np.all(np.concatenate([s.colinds for s in shards]) == csr.colinds)
    # all values match
    assert np.all(np.concatenate([s.values for s in shards]) == csr.values)

    # we can reassemble the shards
    csr2 = CSR._assemble_shards(shards)
    assert csr2.nrows == csr.nrows
    assert csr2.ncols == csr.ncols
    assert csr2.nnz == csr.nnz
    assert np.all(csr2.rowptrs == csr.rowptrs)
    assert np.all(csr2.colinds == csr.colinds)
    assert np.all(csr2.values == csr.values)
