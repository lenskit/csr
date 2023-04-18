"""
Wire together the Numba and Python types.
"""

import numpy as np
from numba import types
from numba.extending import overload_method
from numba.experimental import structref

from ._struct import CSRType
from .csr import CSR
from . import _rows, _struct, structure

structref.define_proxy(CSR, CSRType, [
    'nrows', 'ncols', 'nnz',
    'rowptrs', 'colinds', 'values'
])


@overload_method(CSRType, 'row_extent')
def _csr_row_extent(csr, row):
    return _rows.extent


@overload_method(CSRType, 'row')
def _csr_row(csr, row):
    if isinstance(row, types.Integer):
        if csr.has_values:
            return _rows._row_array_vals
        else:
            return _rows._row_array_ones
    elif isinstance(row, types.ArrayCompatible):
        if csr.has_values:
            return _rows._mr_matrix_vals
        else:
            return _rows._mr_matrix_ones


@overload_method(CSRType, 'row_mask')
def _csr_row_mask(csr, row):
    if isinstance(row, types.Integer):
        return _rows._row_array_mask
    elif isinstance(row, types.ArrayCompatible):
        return _rows._mr_matrix_mask


@overload_method(CSRType, 'row_cs')
def _csr_row_cs(csr, row):
    return _rows.cs


@overload_method(CSRType, 'row_vs')
def _csr_row_vs(csr, row):
    if csr.has_values:
        return _rows._vs_vals
    else:
        return _rows._vs_ones


@overload_method(CSRType, 'rowinds')
def _csr_rowinds(csr, row):
    return _rows.all_indices


@overload_method(CSRType, 'transpose')
def _csr_transpose(csr):
    if csr.has_values:
        return lambda csr: structure._transpose_values(csr)
    else:
        return lambda csr: structure._transpose_structure(csr)


@overload_method(CSRType, 'transpose_structure')
def _csr_transpose_structure(csr):
    return lambda csr: structure._transpose_structure(csr)


@overload_method(CSRType, '_e_value')
def _csr_e_value(csr, i):
    def one(csr, i):
        return 1

    def val(csr, i):
        return csr.values[i]

    if csr.has_values:
        return val
    else:
        return one


@overload_method(CSRType, '_required_values')
def _csr_required_values(csr):
    def ones(csr):
        return np.ones(csr.nnz)

    def vals(csr):
        return csr.values

    if csr.has_values:
        return vals
    else:
        return ones


@overload_method(CSRType, '_filter_zeros')
def _csr_filter_zeros(csr):
    if csr.has_values:
        return _struct._filter_zeros
    else:
        def noop(csr):
            pass
        return noop


@overload_method(CSRType, 'multiply')
def _csr_multiply(csr, other, transpose):
    from . import kernel

    def mult(csr, other, transpose):
        ah = kernel.to_handle(csr)
        bh = kernel.to_handle(other)
        if transpose:
            ch = kernel.mult_abt(ah, bh)
        else:
            ch = kernel.mult_ab(ah, bh)

        kernel.release_handle(bh)
        kernel.release_handle(ah)

        result = kernel.from_handle(ch)
        kernel.release_handle(ch)

        result._filter_zeros()
        return result

    return mult


@overload_method(CSRType, 'mult_vec')
def _csr_mult_vec(csr, x):
    from . import kernel

    def m_ax(csr, x):
        ah = kernel.to_handle(csr)
        y = kernel.mult_vec(ah, x)
        kernel.release_handle(ah)

        return y

    return m_ax
