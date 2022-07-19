"""
CSR test utilities.
"""

from collections.abc import Sequence

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
def finite_arrays(draw, shape, dtype=np.float64(), min_value=-1.0e3, max_value=1.0e3, **kwargs):
    dtype = np.dtype(dtype)
    elts = nph.from_dtype(dtype, min_value=min_value, max_value=max_value,
                          allow_infinity=False, allow_nan=False, **kwargs)
    return draw(nph.arrays(dtype, shape, elements=elts))


@st.composite
def csrs(draw, nrows=None, ncols=None, nnz=None, max_nnz=None, max_density=0.5,
         values=None, dtype=['f4', 'f8']):
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
        nnz_ub = int(np.ceil(nrows * ncols * max_density))
        if max_nnz and nnz_ub > max_nnz:
            nnz_ub = max_nnz
        nnz = draw(st.integers(0, nnz_ub))
    elif not isinstance(nnz, int):
        nnz = draw(nnz)

    coo_elts = st.integers(0, nrows * ncols - 1)
    coords = draw(nph.arrays(np.int32, nnz, elements=coo_elts, unique=True))
    rows = np.mod(coords, nrows, dtype=np.int32)
    cols = np.floor_divide(coords, nrows, dtype=np.int32)

    if isinstance(dtype, st.SearchStrategy):
        dtype = draw(dtype)
    elif isinstance(dtype, Sequence) and not isinstance(dtype, str):
        dtype = draw(st.sampled_from(dtype))
    dtype = np.dtype(dtype)

    if values is None:
        values = draw(st.booleans())
    if values:
        sn = False if values == 'normal' else True
        vals = draw(finite_arrays(nnz, dtype=dtype, allow_subnormal=sn))
        nz = vals != 0.0
        rows = rows[nz]
        cols = cols[nz]
        vals = vals[nz]
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
def mm_pairs(draw, max_shape=(100, 100, 100), dtype=np.float64(), **kwargs):
    "Draw multipliable pairs of matrices"
    mr, mm, mc = max_shape
    rows = draw(st.integers(1, mr))
    mids = draw(st.integers(1, mm))
    cols = draw(st.integers(1, mc))

    if 'values' not in kwargs:
        kwargs = dict(kwargs)
        kwargs['values'] = True

    A = draw(csrs(rows, mids, dtype=dtype, **kwargs))
    B = draw(csrs(mids, cols, dtype=dtype, **kwargs))

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
