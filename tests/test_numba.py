from textwrap import dedent
from _pytest.python_api import approx
import numpy as np
from numpy.testing._private.utils import assert_equal
import scipy.sparse as sps

from csr import CSR
from csr.test_utils import csrs, matrices, mm_pairs, sparse_matrices

from numba import njit

from pytest import raises
from hypothesis import given, assume, settings, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph


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


@given(mm_pairs(), st.booleans())
def test_numba_mult(pair, transpose):
    A, B = pair
    C = A @ B

    A = CSR.from_scipy(A)
    B = CSR.from_scipy(B)

    if transpose:
        B = B.transpose()

    res = _mult(A, B, transpose)

    cnr, cnc = C.shape
    assert res.nrows == cnr
    assert res.ncols == cnc
    assert res.nnz == C.nnz


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
