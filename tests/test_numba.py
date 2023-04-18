"""
Tests to make sure the Numba interface for CSR is usable.
"""

import logging
from textwrap import dedent
from _pytest.python_api import approx
import numpy as np

from csr.test_utils import csrs, mm_pairs

from numba import njit

from hypothesis import given, assume, settings, Phase
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph

_log = logging.getLogger(__name__)
_getters = {}

for __a in ['ncols', 'nrows', 'nnz', 'rowptrs', 'colinds', 'values']:
    __env = {'njit': njit, 'defs': {}}
    exec(dedent(f'''
    @njit
    def _get_{__a}(csr):
        return csr.{__a}
    defs['getter'] = _get_{__a}
    '''), __env)
    _getters[__a] = __env['defs']['getter']


def _get_attr(csr, name):
    getter = _getters[name]
    return getter(csr)


@given(csrs())
def test_access_fields(csr):
    assert _get_attr(csr, 'nrows') == csr.nrows
    assert _get_attr(csr, 'ncols') == csr.ncols
    assert _get_attr(csr, 'nnz') == csr.nnz
    assert all(_get_attr(csr, 'rowptrs') == csr.rowptrs)
    assert all(_get_attr(csr, 'colinds') == csr.colinds)
    if csr.values is None:
        assert _get_attr(csr, 'values') is None
    else:
        assert all(_get_attr(csr, 'values') == csr.values)


@njit
def _extent(csr, row):
    return csr.row_extent(row)


@given(csrs())
def test_csr_row_extent(csr):

    for i in range(csr.nrows):
        sp, ep = _extent(csr, i)
        assert sp == csr.rowptrs[i]
        assert ep == csr.rowptrs[i + 1]


@njit
def _row(csr, row):
    return csr.row(row)


@given(csrs())
def test_csr_row(csr):
    for i in range(csr.nrows):
        cr = csr.row(i)
        nr = _row(csr, i)
        assert nr.shape == cr.shape
        assert all(nr == cr)


@njit
def _rows(csr, rows):
    return csr.row(rows)


@given(st.data(), csrs())
def test_csr_rows(data, csr):
    rows = data.draw(st.lists(st.integers(0, csr.nrows - 1), unique=True))
    rows = np.asarray(rows, dtype=np.int32)
    cmat = csr.row(rows)
    nmat = _rows(csr, rows)

    assert nmat.shape == cmat.shape
    assert np.all(nmat == cmat)


@njit
def _row_mask(csr, rows):
    return csr.row_mask(rows)


@given(st.data(), csrs())
def test_csr_row_mask(data, csr):
    csr = data.draw(csrs(st.integers(1, 100), st.integers(1, 100)))

    row_id = st.integers(0, csr.nrows - 1)
    row_list = st.lists(row_id, unique=True)
    rows = data.draw(st.one_of(row_id, row_list))
    if isinstance(rows, list):
        rows = np.asarray(rows, dtype=np.int32)

    nr_mask = _row_mask(csr, rows)
    assert nr_mask.dtype == np.bool_
    cr_mask = csr.row_mask(rows)
    assert nr_mask.shape == cr_mask.shape
    assert np.all(nr_mask == cr_mask)


@njit
def _row_cvs(csr, row):
    cs = csr.row_cs(row)
    vs = csr.row_vs(row)
    return cs, vs


@given(csrs())
def test_csr_row_cvs(csr):
    for i in range(csr.nrows):
        ccs = csr.row_cs(i)
        cvs = csr.row_vs(i)
        ncs, nvs = _row_cvs(csr, i)
        assert ncs.shape == ccs.shape
        assert nvs.shape == cvs.shape
        assert all(ncs == ccs)
        assert all(nvs == cvs)


@njit
def _transpose(csr):
    return csr.transpose()


@given(csrs())
def test_transpose(csr):
    t = _transpose(csr)

    assert t.nrows == csr.ncols
    assert t.ncols == csr.nrows
    assert t.nnz == csr.nnz

    t2 = csr.transpose()
    assert all(t.rowptrs == t2.rowptrs)
    assert all(t.colinds == t2.colinds)

    if csr.values is not None:
        assert t.values is not None
        assert all(t.values == t2.values)
    else:
        assert t.values is None


@njit
def _transpose_s(csr):
    return csr.transpose_structure()


@given(csrs())
def test_transpose_structure(csr):
    t = _transpose_s(csr)

    assert t.nrows == csr.ncols
    assert t.ncols == csr.nrows
    assert t.nnz == csr.nnz

    t2 = csr.transpose()
    assert all(t.rowptrs == t2.rowptrs)
    assert all(t.colinds == t2.colinds)

    assert t.values is None


@njit
def _mult(A, B, transpose):
    return A.multiply(B, transpose)


def isnormal(x):
    finf = np.finfo(x.dtype)
    mask = x == 0
    mask |= np.abs(x) >= finf.tiny
    return mask


# @settings(phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.explain])
@given(mm_pairs(values='normal'), st.booleans())
def test_numba_mult(pair, transpose):
    A, B = pair
    assume(np.all(isnormal(A.values)))
    assume(np.all(isnormal(B.values)))

    dA = A.to_scipy().toarray()
    dB = B.to_scipy().toarray()
    dC = dA @ dB

    if transpose:
        B = B.transpose()

    res = _mult(A, B, transpose)

    cnr, cnc = dC.shape
    assert res.nrows == cnr
    assert res.ncols == cnc
    try:
        assert res.to_scipy().toarray() == approx(dC, rel=1.0e-5, abs=1.0e-10)
    except AssertionError as e:
        # let's do a little diagnostic
        rnp = res.to_scipy().toarray()
        mask = rnp != dC
        _log.info('CSR where diff: %s', rnp[mask])
        _log.info('numpy where diff: %s', dC[mask])
        (nzr, nzc) = mask.nonzero()
        for r, c in zip(nzr, nzc):
            if dC[r, c] == 0:
                _log.info('should be 0, is %e', rnp[r, c])
                _log.info('left row:\n%s', dA[r, :])
                _log.info('right col:\n%s', dB[:, c])
                _log.info('dot prod is %e', np.dot(dA[r, :], dB[:, c]))
        raise e


@njit
def _mult_vec(A, x):
    return A.mult_vec(x)


@given(st.data())
def test_numba_mult_vec(data):
    A = data.draw(csrs())
    vals = st.floats(-100, 100)
    x = data.draw(nph.arrays(np.float64, A.ncols, elements=vals))

    y = _mult_vec(A, x)

    assert y.shape == (A.nrows,)
    assert y == approx(A.to_scipy() @ x, nan_ok=True, rel=1.0e-5)
