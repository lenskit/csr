from textwrap import dedent
import numpy as np
import scipy.sparse as sps

from csr import CSR
from csr.test_utils import csrs, matrices, sparse_matrices

from numba import njit

from pytest import raises
from hypothesis import given, assume, settings, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph


_getters = {}

for __a in ['ncols', 'nrows', 'nnz', 'rowptrs', 'colinds', 'has_values', 'values']:
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
        assert not _get_attr(csr, 'has_values')
        assert _get_attr(csr, 'values').size == 0
    else:
        assert _get_attr(csr, 'has_values')
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
