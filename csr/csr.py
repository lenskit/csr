"""
Python API for CSR matrices.
"""

import warnings
import numpy as np
import scipy.sparse as sps

from numba import njit
from numba.core import types
from numba.extending import overload_attribute, overload_method
from numba.experimental import structref

from csr.kernels import get_kernel, releasing

####################################################
#region CSR type and attribute accessors


@structref.register
class CSRType(types.StructRef):
    pass


@njit
def _csr_get_nrows(self):
    return self.nrows


@njit
def _csr_get_ncols(self):
    return self.ncols


@njit
def _csr_get_nnz(self):
    return self.nnz


@njit
def _csr_get_rowptrs(self):
    return self.rowptrs


@njit
def _csr_get_colinds(self):
    return self.colinds


@njit
def _csr_get_values(self):
    return self.values


@njit
def _csr_set_values(self, values):
    self.values = values
    self.has_values = True


@njit
def _csr_clear_values(self):
    self.values = np.zeros(0)
    self.has_values = False


@njit
def _csr_has_values(self):
    return self.has_values

#endregion

####################################################
#region CSR access functions


def _row_extent(csr, row):
    "Get the extent of a row in the matrix storage."
    sp = csr.rowptrs[row]
    ep = csr.rowptrs[row + 1]
    return sp, ep


def _row(csr, row):
    "Get a row as a dense vector."
    v = np.zeros(csr.ncols)
    if csr.nnz == 0:
        return v

    sp, ep = csr.row_extent(row)
    cols = csr.colinds[sp:ep]
    if csr.has_values > 0:
        v[cols] = csr.values[sp:ep]
    else:
        v[cols] = 1

    return v


def _row_cs(csr, row):
    "Get the column indices for a row."
    sp, ep = csr.row_extent(row)
    return csr.colinds[sp:ep]


def _row_vs(csr, row):
    "Get the nonzero values for a row."
    sp, ep = csr.row_extent(row)

    if csr.has_values:
        return csr.values[sp:ep]
    else:
        return np.full(ep - sp, 1.0)


def _rowinds(csr):
    "Get the row indices for the nonzero values in a matrix."
    ris = np.zeros(csr.nnz, np.intc)
    for i in range(csr.nrows):
        sp, ep = csr.row_extent(i)
        ris[sp:ep] = i
    return ris


#endregion

###################################################
#region CSR class


class CSR(structref.StructRefProxy):
    """
    Simple compressed sparse row matrix.  This is like :py:class:`scipy.sparse.csr_matrix`, with
    a couple of useful differences:

    * The value array is optional, for cases in which only the matrix structure is required.
    * The value array, if present, is always double-precision.

    You generally don't want to create this class yourself with the constructor.  Instead, use one
    of its class or static methods.

    This class, with its attributes and several of its methods, is also usable from Numba (it is a
    proxy for a Numba StructRef).  When used from Numba, :py:attr:`values` is always present, but
    has length 0 when there is no value array.  Use the :py:attr:`has_values` attribute to check
    for its presence.

    Attributes:
        nrows(int): the number of rows.
        ncols(int): the number of columns.
        nnz(int): the number of entries.
        rowptrs(numpy.ndarray): the row pointers.
        colinds(numpy.ndarray): the column indices.
        has_values(bool): whether the array has values.
        values(numpy.ndarray or None): the values.
    """

    def __new__(cls, nrows, ncols, nnz, ptrs, inds, vals):
        if vals is None:
            hv = False
            vals = np.zeros(0)
        else:
            hv = True

        return structref.StructRefProxy.__new__(cls, nrows, ncols, nnz, ptrs, inds, hv, vals)

    @classmethod
    def empty(cls, nrows, ncols, row_nnzs=None):
        """
        Create an uninitialized CSR matrix.

        Args:
            nrows(int): the number of rows.
            ncols(int): the number of columns.
            row_nnzs(array-like):
                the number of nonzero entries for each row, or None for an empty matrix.
        """
        from .constructors import create_empty, create_from_sizes
        if row_nnzs is not None:
            assert len(row_nnzs) == nrows
            return create_from_sizes(nrows, ncols, row_nnzs)
        else:
            return create_empty(nrows, ncols)

    @classmethod
    def from_coo(cls, rows, cols, vals, shape=None, *, rpdtype=np.intc):
        """
        Create a CSR matrix from data in COO format.

        Args:
            rows(array-like): the row indices.
            cols(array-like): the column indices.
            vals(array-like): the data values; can be ``None``.
            shape(tuple): the array shape, or ``None`` to infer from row & column indices.
        """
        from .structure import from_coo
        if shape is not None:
            nrows, ncols = shape
            assert np.max(rows, initial=0) < nrows
            assert np.max(cols, initial=0) < ncols
        else:
            nrows = np.max(rows) + 1
            ncols = np.max(cols) + 1

        nnz = len(rows)
        assert len(cols) == nnz
        assert vals is None or len(vals) == nnz

        rowptrs, cols, vals = from_coo(nrows, rows, cols, vals)
        return cls(nrows, ncols, nnz, rowptrs, cols, vals)

    @classmethod
    def from_scipy(cls, mat, copy=True):
        """
        Convert a scipy sparse matrix to a CSR.

        Args:
            mat(scipy.sparse.spmatrix): a SciPy sparse matrix.
            copy(bool): if ``False``, reuse the SciPy storage if possible.

        Returns:
            CSR: a CSR matrix.
        """
        if not sps.isspmatrix_csr(mat):
            mat = mat.tocsr(copy=copy)
        rp = np.require(mat.indptr, np.intc, 'C')
        if copy and rp is mat.indptr:
            rp = rp.copy()
        cs = np.require(mat.indices, np.intc, 'C')
        if copy and cs is mat.indices:
            cs = cs.copy()
        vs = mat.data.copy() if copy else mat.data
        return cls(mat.shape[0], mat.shape[1], mat.nnz, rp, cs, vs)

    def to_scipy(self):
        """
        Convert a CSR matrix to a SciPy :py:class:`scipy.sparse.csr_matrix`.  Avoids copying
        if possible.

        Args:
            self(CSR): A CSR matrix.

        Returns:
            scipy.sparse.csr_matrix:
                A SciPy sparse matrix with the same data.
        """
        values = self.values
        if values is None:
            values = np.full(self.nnz, 1.0)
        return sps.csr_matrix((values, self.colinds, self.rowptrs), shape=(self.nrows, self.ncols))

    nrows = property(_csr_get_nrows)
    ncols = property(_csr_get_ncols)
    nnz = property(_csr_get_nnz)
    rowptrs = property(_csr_get_rowptrs)
    colinds = property(_csr_get_colinds)
    has_values = property(_csr_has_values)

    @property
    def values(self):
        return _csr_get_values(self) if _csr_has_values(self) else None

    @values.setter
    def values(self, vs: np.ndarray):
        if vs is None:
            _csr_clear_values(self)
        else:
            if len(vs) < self.nnz:
                raise ValueError('value array too small')
            elif len(vs) > self.nnz:
                vs = vs[:self.nnz]
            _csr_set_values(self, vs)

    @property
    def R(self):
        warnings.warn('.R deprecated, use CSR directly', DeprecationWarning)
        return self

    def copy(self, include_values=True, *, copy_structure=True):
        """
        Create a copy of this CSR.

        Args:
            include_values(bool): whether to copy the values or only the structure.
            copy_structure(bool):
                whether to copy the structure (index & pointers) or share with the original matrix.
        """
        values = self.values
        if include_values and values is not None:
            values = np.copy(values)
        else:
            values = None
        rps = self.rowptrs
        cis = self.colinds
        if copy_structure:
            rps = np.copy(rps)
            cis = np.copy(cis)
        return CSR(self.nrows, self.ncols, self.nnz,
                   rps, cis, values)

    def sort_rows(self):
        """
        Sort the rows of this matrix in column order.  This is an **in-place operation**.
        """
        from .structure import sort_rows
        sort_rows(self)

    def subset_rows(self, begin, end):
        """
        Subset the rows in this matrix.
        """
        from .structure import subset_rows
        return subset_rows(self, begin, end)

    def rowinds(self) -> np.ndarray:
        """
        Get the row indices from this array.  Combined with :py:attr:`colinds` and
        :py:attr:`values`, this can form a COO-format sparse matrix.
        """
        return _rowinds(self)

    def row(self, row):
        """
        Return a row of this matrix as a dense ndarray.

        Args:
            row(int): the row index.

        Returns:
            numpy.ndarray:
                the row, with 0s in the place of missing values.  If the CSR only
                stores matrix structure, the returned vector has 1s where the CSR
                records an entry.
        """
        return _row(self, row)

    def row_extent(self, row):
        """
        Get the extent of a row in the underlying column index and value arrays.

        Args:
            row(int): the row index.

        Returns:
            tuple: ``(s, e)``, where the row occupies positions :math:`[s, e)` in the
            CSR data.
        """
        return _row_extent(self, row)

    def row_cs(self, row):
        """
        Get the column indcies for the stored values of a row.
        """
        return _row_cs(self, row)

    def row_vs(self, row):
        """
        Get the stored values of a row.  If only the matrix structure is stored, this
        returns a vector of 1s.
        """
        return _row_vs(self, row)

    def row_nnzs(self):
        """
        Get a vector of the number of nonzero entries in each row.

        Returns:
            numpy.ndarray: the number of nonzero entries in each row.
        """
        return np.diff(self.rowptrs)

    def normalize_rows(self, normalization):
        """
        Normalize the rows of the matrix.

        .. note:: The normalization *ignores* missing values instead of treating
                  them as 0.

        Args:
            normalization(str):
                The normalization to perform. Can be one of:

                * ``'center'`` - center rows about the mean
                * ``'unit'`` - convert rows to a unit vector

        Returns:
            numpy.ndarray:
                The normalization values for each row.
        """
        if normalization == 'center':
            return _ops.center_rows(self)
        elif normalization == 'unit':
            return _ops.unit_rows(self)
        else:
            raise ValueError('unknown normalization: ' + normalization)

    def transpose(self, include_values=True):
        """
        Transpose a CSR matrix.

        Args:
            include_values(bool): whether to include the values in the transpose.

        Returns:
            CSR: the transpose of this matrix (or, equivalently, this matrix in CSC format).
        """
        from .structure import transpose
        return transpose(self, include_values)

    def filter_nnzs(self, filt):
        """
        Filter the values along the full NNZ axis.

        Args:
            filt(ndarray):
                a logical array of length :attr:`nnz` that indicates the values to keep.

        Returns:
            CSR: The filtered sparse matrix.
        """
        if len(filt) != self.nnz:
            raise ValueError('filter has length %d, expected %d' % (len(filt), self.nnz))
        rps2 = np.zeros_like(self.rowptrs)
        for i in range(self.nrows):
            sp, ep = self.row_extent(i)
            rlen = np.sum(filt[sp:ep])
            rps2[i + 1] = rps2[i] + rlen

        nnz2 = rps2[-1]
        assert nnz2 == np.sum(filt)

        cis2 = self.colinds[filt]
        vs = self.values
        vs2 = None if vs is None else vs[filt]

        return CSR(self.nrows, self.ncols, nnz2, rps2, cis2, vs2)

    def multiply(self, other, *, transpose=False):
        """
        Multiply this matrix by another.

        Args:
            other(CSR): the other matrix.
            transpose(bool): if ``True``, compute :math:`AB^{T}` instead of :math:`AB`.

        Returns
            CSR: the product of the two matrices.
        """
        if transpose:
            assert self.ncols == other.ncols
        else:
            assert self.ncols == other.nrows

        K = get_kernel()
        with releasing(K.to_handle(self), K) as a_h:
            with releasing(K.to_handle(other), K) as b_h:
                if transpose:
                    c_h = K.mult_abt(a_h, b_h)
                else:
                    c_h = K.mult_ab(a_h, b_h)
                with releasing(c_h, K):
                    crepr = K.from_handle(c_h)

        return crepr

    def mult_vec(self, v):
        """
        Multiply this matrix by a vector.

        Args:
            other(numpy.ndarray): A vector, of length `ncols`.

        Returns:
            numpy.ndarray: :math:`A\\vec{x}`, as a vector.
        """
        assert v.shape == (self.ncols,)
        K = get_kernel()
        with releasing(K.to_handle(self), K) as h:
            return K.mult_vec(h, v)

    def drop_values(self):
        """
        Remove the value array from this CSR.  This is an **in-place** operation.
        """
        _csr_clear_values(self)

    def fill_values(self, value):
        """
        Fill the values of this CSR with the specified value.  If the CSR is
        structure-only, a value array is added.  This is an **in-place** operation.
        """
        _csr_set_values(self, np.full(self.nnz, value, dtype='float64'))

    def __str__(self):
        return '<CSR {}x{} ({} nnz)>'.format(self.nrows, self.ncols, self.nnz)

    def __repr__(self):
        repr = '<CSR {}x{} ({} nnz)'.format(self.nrows, self.ncols, self.nnz)
        repr += ' {\n'
        repr += '  rowptrs={}\n'.format(self.rowptrs)
        repr += '  colinds={}\n'.format(self.colinds)
        repr += '  values={}\n'.format(self.values)
        repr += '}'
        return repr

    def __reduce__(self):
        args = (self.nrows, self.ncols, self.nnz, self.rowptrs, self.colinds, self.values)
        return (CSR, args)

#endregion

###############################################
#region Numba struct setup and methods


structref.define_proxy(CSR, CSRType, [
    'nrows', 'ncols', 'nnz',
    'rowptrs', 'colinds',
    'has_values', 'values'
])


@overload_method(CSRType, 'row_extent')
def _csr_row_extent(csr, row):
    return _row_extent


@overload_method(CSRType, 'row')
def _csr_row(csr, row):
    return _row


@overload_method(CSRType, 'row_cs')
def _csr_row_cs(csr, row):
    return _row_cs


@overload_method(CSRType, 'row_vs')
def _csr_row_vs(csr, row):
    return _row_vs


@overload_method(CSRType, 'rowinds')
def _csr_rowinds(csr, row):
    return _rowinds


@overload_method(CSRType, 'transpose')
def _csr_transpose(csr, include_values):
    from .structure import transpose
    return transpose

#endregion
