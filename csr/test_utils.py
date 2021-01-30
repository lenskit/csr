"""
CSR test utilities.
"""

import numpy as np
import scipy.sparse as sps

from hypothesis import settings, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph

from .csr import CSR


def fractions(**kwargs):
    return st.floats(0, 1, **kwargs)


@st.composite
def csrs(draw, nrows=None, ncols=None, nnz=None, values=None):
    "Draw CSR matrices by generating COO data."
    if ncols is None:
        ncols = draw(st.integers(5, 100))
    elif not isinstance(ncols, int):
        ncols = draw(ncols)

    if nrows is None:
        nrows = draw(st.integers(5, 100))
    elif not isinstance(nrows, int):
        nrows = draw(nrows)

    if nnz is None:
        nnz = draw(st.integers(10, nrows * ncols // 2))
    elif not isinstance(nnz, int):
        nnz = draw(nnz)

    coords = draw(nph.arrays(np.int32, nnz, elements=st.integers(0, nrows*ncols - 1), unique=True))
    rows = np.mod(coords, nrows, dtype=np.int32)
    cols = np.floor_divide(coords, nrows, dtype=np.int32)
    if values is None:
        values = draw(st.booleans())
    if values:
        rng = draw(st.randoms())
        vals = np.array([rng.normalvariate(0, 1) for i in range(nnz)])
    else:
        vals = None
    return CSR.from_coo(rows, cols, vals, (nrows, ncols))


@st.composite
def sparse_matrices(draw, max_shape=(1000, 1000), density=fractions(exclude_min=True), format='csr'):
    ubr, ubc = max_shape
    rows = draw(st.integers(1, ubr))
    cols = draw(st.integers(1, ubc))
    dens = draw(density)
    return sps.random(rows, cols, dens, format=format)


def matrices(max_shape=(100, 100), dtype='f8'):
    "Draw dense matrices"
    ubr, ubc = max_shape
    return nph.arrays(dtype, st.tuples(st.integers(1, ubr), st.integers(1, ubc)))


csr_slow = settings(deadline=None, suppress_health_check=HealthCheck.all(), max_examples=15)
