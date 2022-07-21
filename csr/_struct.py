"""
Base definitions of CSR structure.
"""

from numba import njit
from numba.core import types
from numba.experimental import structref


@structref.register
class CSRType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

    @property
    def has_values(self):
        "Query whether this CSR type has values."
        return not isinstance(self.field_dict['values'], types.NoneType)

    @property
    def ptr_type(self):
        "The type of pointers into the data."
        return self.field_dict['rowptrs']

    @property
    def val_type(self):
        "The value type for the CSR"
        return self.field_dict['values']


@njit
def get_nrows(self):
    return self.nrows


@njit
def get_ncols(self):
    return self.ncols


@njit
def get_nnz(self):
    return self.nnz


@njit
def get_rowptrs(self):
    return self.rowptrs


@njit
def get_colinds(self):
    return self.colinds


@njit
def get_values(self):
    return self.values


def _filter_zeros(csr):
    "Filter out the zero values in a CSR, in-place. Only works when CSR has values."
    nnz = 0
    for i in range(csr.nrows):
        sp, ep = csr.row_extent(i)
        csr.rowptrs[i] = nnz
        for jp in range(sp, ep):
            if csr.values[jp] != 0:
                csr.colinds[nnz] = csr.colinds[jp]
                csr.values[nnz] = csr.values[jp]
                nnz += 1

    csr.rowptrs[csr.nrows] = nnz
    csr.nnz = nnz
    csr.colinds = csr.colinds[:nnz]
    csr.values = csr.values[:nnz]


filter_zeros = njit(nogil=True)(_filter_zeros)
