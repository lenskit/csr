"""
Internal tests for Numba kernel multiplication code.
"""

import numpy as np
import scipy.sparse as sps

from csr import CSR
from csr.test_utils import csrs, csr_slow, sparse_matrices, mm_pairs
from csr.kernels.numba.multiply import _sym_mm

from pytest import mark, approx, raises
from hypothesis import given, assume, settings, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph


@settings(deadline=None)
@given(mm_pairs(max_shape=(50, 1000, 50), as_csr=True))
def test_symb(pair):
    A, B = pair

    cp = np.zeros_like(A.rowptrs)

    cci = _sym_mm(A.R, B.R, cp)

    # Is everything in range?
    assert all(cci >= 0)
    assert all(cci < B.ncols)

    # Are column pointers nondecreasing?
    assert all(np.diff(cp) >= 0)

    # Do we have the right number of NNZs?
    assert len(cci) == cp[A.nrows]
