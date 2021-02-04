"""
Type for CSR storage.
"""

import numpy as np
from numba.experimental import jitclass
import numba.types as nt


@jitclass([
    ('nrows', nt.intc),
    ('ncols', nt.intc),
    ('nnz', nt.intc),
    ('has_values', nt.boolean),
    ('rowptrs', nt.intc[::1]),
    ('colinds', nt.intc[::1]),
    ('values', nt.float64[::1])
])
class _CSR:
    """
    Internal storage for :py:class:`csr.CSR`. If you work with CSRs from Numba,
    you will use this type instead of the CSR class itself, with functions
    from :py:mod:`csr.native_ops`.

    This has the same key attributes as :py:class:`csr.CSR`, except that it always
    has a ``values`` array; if only structure is stored, this array has length 0.

    Attributes:
        nrows(int): the number of rows
        ncols(int): the number of columns
        nnz(int): the number of nonzero entries
        rowptrs(numpy.ndarray): starting position of each row (length ``nrows + 1``)
        colinds(numpy.ndarray): column indices (length ``nnz``)
        values(numpy.ndarray):
            matrix cell values (length ``nnz`` or 0). If only the matrix structure
            is stored, this has length 0.
        has_values(bool):
            Whether or not this matrix has values.
    """
    def __init__(self, nrows, ncols, nnz, rowptrs, colinds, values):
        self.nrows = nrows
        self.ncols = ncols
        self.nnz = nnz
        self.rowptrs = rowptrs
        self.colinds = colinds
        if values is None or values.size == 0:
            self.values = EMPTY_VALUES
            self.has_values = False
        else:
            self.values = values
            self.has_values = True


EMPTY_VALUES = np.zeros(0)
