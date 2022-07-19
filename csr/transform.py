"""
Data transformation implementations.
"""

import logging
import numpy as np
import math
from numba import njit

_log = logging.getLogger(__name__)


@njit(nogil=True)
def center_rows(csr):
    "Mean-center the nonzero values of each row of a CSR."
    means = np.zeros(csr.nrows, dtype=csr.values.dtype)
    for i in range(csr.nrows):
        sp, ep = csr.row_extent(i)
        if sp == ep:
            continue  # empty row
        vs = csr.row_vs(i)
        m = np.mean(vs)
        means[i] = m
        csr.values[sp:ep] -= m

    return means


@njit(nogil=True)
def unit_rows(csr):
    "Normalize the rows of a CSR to unit vectors."
    info = np.finfo(csr.values.dtype)
    norms = np.zeros(csr.nrows, dtype=csr.values.dtype)
    for i in range(csr.nrows):
        sp, ep = csr.row_extent(i)
        if sp == ep:
            continue  # empty row
        vs = csr.row_vs(i)

        # one would think that unit-normalizing is a straightforward operation, but IEEE-754
        # disagrees.  If the values are all very small, the naive  approach will result in an
        # incorrect norm (because some of the squares will be zero or unhelpfully close to it),
        # and the resulting row will have a norm that exceeds 1.
        #
        # This solution is courtesy of @jekstrand:
        # https://twitter.com/jekstrand_/status/1549222506938130433
        #
        # We use a pre-normalization stage to get the values up into a reasonable range, before
        # computing the proper Euclidean norm.

        # first, find the maximum absolute value
        vmax = np.max(np.abs(vs))

        # get its exponent, and pre-normalize to bring the values up if they're all tiny
        vm, ve = math.frexp(vmax)
        pnexp = min(-ve, info.maxexp - 1)
        pnexp = max(pnexp, info.minexp)
        prenorm = math.ldexp(1.0, pnexp)
        csr.values[sp:ep] *= prenorm

        # now we normalize the reasonably-scaled values
        inorm = np.linalg.norm(csr.values[sp:ep])
        norms[i] = inorm / prenorm
        csr.values[sp:ep] /= inorm

    return norms
