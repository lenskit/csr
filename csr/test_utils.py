"""
CSR test utilities.
"""

import numpy as np
import scipy.sparse as sps

import psutil
from hypothesis import settings, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as nph

from .csr import CSR


def fractions(**kwargs):
    return st.floats(0, 1, **kwargs)


@st.composite
def finite_arrays(draw, shape, dtype=np.float64(), min_value=1.0e-6, max_value=1.0e6):
    dtype = np.dtype(dtype)
    elts = nph.from_dtype(dtype, min_value=min_value, max_value=max_value,
                          allow_infinity=False, allow_nan=False)
    return draw(nph.arrays(dtype, shape, elements=elts))


@st.composite
def csrs(draw, nrows=None, ncols=None, nnz=None, values=None, dtype=np.float64()):
    "Draw CSR matrices by generating COO data."
    if ncols is None:
        ncols = draw(st.integers(1, 80))
    elif not isinstance(ncols, int):
        ncols = draw(ncols)

    if nrows is None:
        nrows = draw(st.integers(1, 80))
    elif not isinstance(nrows, int):
        nrows = draw(nrows)

    if nnz is None:
        nnz = draw(st.integers(0, int(np.ceil(nrows * ncols * 0.5))))
    elif not isinstance(nnz, int):
        nnz = draw(nnz)

    coords = draw(nph.arrays(np.int32, nnz, elements=st.integers(0, nrows*ncols - 1), unique=True))
    rows = np.mod(coords, nrows, dtype=np.int32)
    cols = np.floor_divide(coords, nrows, dtype=np.int32)

    if isinstance(dtype, st.SearchStrategy):
        dtype = draw(dtype)
    elif isinstance(dtype, list):
        dtype = draw(st.sampled_from(dtype))
    dtype = np.dtype(dtype)

    if values is None:
        values = draw(st.booleans())
    if values:
        vals = draw(finite_arrays(nnz, dtype=dtype))
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
def mm_pairs(draw, max_shape=(100, 100, 100), dtype=np.float64()):
    "Draw multipliable pairs of matrices"
    mr, mm, mc = max_shape
    rows = draw(st.integers(1, mr))
    mids = draw(st.integers(1, mm))
    cols = draw(st.integers(1, mc))

    A = draw(csrs(rows, mids, values=True, dtype=dtype))
    B = draw(csrs(mids, cols, values=True, dtype=dtype))

    return A, B


def matrices(max_shape=(100, 100), dtype='f8'):
    "Draw dense matrices"
    ubr, ubc = max_shape
    return nph.arrays(dtype, st.tuples(st.integers(1, ubr), st.integers(1, ubc)))


def csr_slow(divider=2):
    dft = settings.default
    return settings(dft, deadline=None, suppress_health_check=HealthCheck.all(),
                    max_examples=dft.max_examples // divider)


def has_memory(req_gb=32):
    req_bytes = req_gb * 1024 * 1024 * 1024
    vm = psutil.virtual_memory()
    return vm.total >= req_bytes
