"""
Type for CSR storage.
"""

from collections import namedtuple
import numpy as np

_CSR = namedtuple('_CSR', [
    'nrows', 'ncols', 'nnz',
    'rowptrs', 'colinds', 'values'
], module=__name__)
_CSR.__doc__ = """
Internal storage for :py:class:`csr.CSR`. If you work with CSRs from Numba,
you will use this namedtuple instead of the CSR class itself, with functions
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
"""

EMPTY_VALUES = np.zeros(0)
