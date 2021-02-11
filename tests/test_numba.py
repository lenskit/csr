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


def _get_attr(csr, name):
    "Helper to synthesize Numba-compiled functions that access attributes."
    env = {'njit': njit, 'defs': {}}
    exec(dedent(f'''
    @njit
    def _get_{name}(csr):
        return csr.{name}
    defs['getter'] = _get_{name}
    '''), env)
    getter = env['defs']['getter']
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


@given(csrs())
def test_csr_row_extent(csr):
    @njit
    def _extent(csr, row):
        return csr.row_extent(row)

    for i in range(csr.nrows):
        sp, ep = _extent(csr, i)
        assert sp == csr.rowptrs[i]
        assert ep == csr.rowptrs[i+1]
