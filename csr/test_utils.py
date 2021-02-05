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
        ncols = draw(st.integers(1, 100))
    elif not isinstance(ncols, int):
        ncols = draw(ncols)

    if nrows is None:
        nrows = draw(st.integers(1, 100))
    elif not isinstance(nrows, int):
        nrows = draw(nrows)

    if nnz is None:
        nnz = draw(st.integers(0, int(np.ceil(nrows * ncols * 0.5))))
    elif not isinstance(nnz, int):
        nnz = draw(nnz)

    coords = draw(nph.arrays(np.int32, nnz, elements=st.integers(0, nrows*ncols - 1), unique=True))
    rows = np.mod(coords, nrows, dtype=np.int32)
    cols = np.floor_divide(coords, nrows, dtype=np.int32)
    if values is None:
        values = draw(st.booleans())
    if values:
        vals = draw(nph.arrays(np.float64, nnz, elements=st.floats(-10, 10)))
    else:
        vals = None
    return CSR.from_coo(rows, cols, vals, (nrows, ncols))


@st.composite
def sparse_matrices(draw, max_shape=(1000, 1000), density=fractions(), format='csr'):
    ubr, ubc = max_shape
    rows = draw(st.integers(1, ubr))
    cols = draw(st.integers(1, ubc))
    dens = draw(density)
    return sps.random(rows, cols, dens, format=format)


@st.composite
def mm_pairs(draw, max_shape=(100, 100, 100), as_csr=False):
    "Draw multipliable pairs of matrices"
    mr, mm, mc = max_shape
    rows = draw(st.integers(1, mr))
    mids = draw(st.integers(1, mm))
    cols = draw(st.integers(1, mc))
    dA = draw(st.floats(0.001, 0.9))
    dB = draw(st.floats(0.001, 0.9))

    A = sps.random(rows, mids, dA, format='csr')
    B = sps.random(mids, cols, dB, format='csr')

    if as_csr:
        return CSR.from_scipy(A), CSR.from_scipy(B)
    else:
        return A, B


def matrices(max_shape=(100, 100), dtype='f8'):
    "Draw dense matrices"
    ubr, ubc = max_shape
    return nph.arrays(dtype, st.tuples(st.integers(1, ubr), st.integers(1, ubc)))


def csr_slow(divider=2):
    dft = settings.default
    return settings(dft, deadline=None, suppress_health_check=HealthCheck.all(),
                    max_examples=dft.max_examples // divider)
