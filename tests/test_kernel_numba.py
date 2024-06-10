"""
Internal tests for Numba kernel multiplication.
"""

import numpy as np

from csr.test_utils import csr_slow, mm_pairs
from csr.kernels.numba.multiply import _sym_mm

from hypothesis import given


@csr_slow()
@given(mm_pairs(max_shape=(50, 200, 50)))
def test_symb(pair):
    A, B = pair

    cp = np.zeros_like(A.rowptrs)

    cci = _sym_mm(A, B, cp)

    # Is everything in range?
    assert all(cci >= 0)
    assert all(cci < B.ncols)

    # Are column pointers nondecreasing?
    assert all(np.diff(cp) >= 0)

    # Do we have the right number of NNZs?
    assert len(cci) == cp[A.nrows]
