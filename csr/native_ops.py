"""
Backend implementations of Numba operations.
"""

import logging
import numpy as np
from numba import njit

from .csr import CSR, _row_extent

_log = logging.getLogger(__name__)


@njit(nogil=True)
def center_rows(csr):
    "Mean-center the nonzero values of each row of a CSR."
    means = np.zeros(csr.nrows)
    for i in range(csr.nrows):
        sp, ep = row_extent(csr, i)
        if sp == ep:
            continue  # empty row
        vs = row_vs(csr, i)
        m = np.mean(vs)
        means[i] = m
        csr.values[sp:ep] -= m

    return means


@njit(nogil=True)
def unit_rows(csr):
    "Normalize the rows of a CSR to unit vectors."
    norms = np.zeros(csr.nrows)
    for i in range(csr.nrows):
        sp, ep = row_extent(csr, i)
        if sp == ep:
            continue  # empty row
        vs = row_vs(csr, i)
        m = np.linalg.norm(vs)
        norms[i] = m
        csr.values[sp:ep] /= m

    return norms
