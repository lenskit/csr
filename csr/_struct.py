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
