"""
Python API for CSR matrices.
"""

import warnings
import numpy as np
import scipy.sparse as sps

from numba import config
from numba.experimental import structref

from csr.kernels import get_kernel, releasing
from . import _struct, _rows

INTC = np.iinfo(np.intc)

# ugly hack for a bug on Numba < 0.53
if config.DISABLE_JIT:
    class _csr_base:
        def __init__(self, nrows, ncols, nnz, ptrs, inds, vals, cast=True):
            self.nrows = nrows
            self.ncols = ncols
            self.nnz = nnz
            if cast and np.max(ptrs) <= INTC.max:
                self.rowptrs = np.require(ptrs, np.intc, 'C')
            else:
                self.rowptrs = np.require(ptrs, requirements='C')
            self.colinds = np.require(inds, np.intc, 'C')
            if vals is not None:
                self._values = np.require(vals, requirements='C')
            else:
                self._values = None

        def _numba_box_(self, *args):
            raise NotImplementedError()

    NUMBA_ENABLED = False

else:
    _csr_base = structref.StructRefProxy
    NUMBA_ENABLED = True


class CSR(_csr_base):
    """
    Simple compressed sparse row matrix.  This is like :py:class:`scipy.sparse.csr_matrix`, with
    a few useful differences:

    * The value array is optional, for cases in which only the matrix structure is required.
    * The value array, if present, is always double-precision.
    * It is usable from code compiled in Numba's nopython mode.

    You generally don't want to create this class yourself with the constructor.  Instead, use one
    of its class or static methods.

    Not all methods are available from Numba, and a few have restricted signatures.  The
    documentation for each method notes deviations when in Numba-compiled code.

    At the Numba level, matrices with and without value arrays have different types. For the
    most part, this is transparent, but if you want to write a Numba function that works on
    the values array but only if it is present, it requires writing two versions of the
    function and using :py:func:`numba.extending.overload` to dispatch to the correct one.
    There are several examples of doing this in the CSR source code. The method
    :py:meth:`CSRType.has_values` lets you quickly see if a CSR type instance has
    values or not.

    Attributes:
        nrows(int): the number of rows.
        ncols(int): the number of columns.
        nnz(int): the number of entries.
        rowptrs(numpy.ndarray): the row pointers.
        colinds(numpy.ndarray): the column indices.
        values(numpy.ndarray or None): the values.
    """

    def __new__(cls, nrows, ncols, nnz, rps, cis, vs, cast=True):
        assert nrows >= INTC.min
        assert nrows <= INTC.max
        assert ncols >= INTC.min
        assert ncols <= INTC.max
        nrows = np.intc(nrows)
        ncols = np.intc(ncols)
        if cast:
            cis = np.require(cis, np.intc, 'C')
            if nnz <= INTC.max:
                rps = np.require(rps, np.intc, 'C')
            if vs is not None:
                vs = np.require(vs, requirements='C')

        if NUMBA_ENABLED:
            return _csr_base.__new__(cls, nrows, ncols, nnz, rps, cis, vs)
        else:
            return _csr_base.__new__(cls)


    @classmethod
    def empty(cls, nrows, ncols, row_nnzs=None, values=True):
        """
        Create an uninitialized CSR matrix.

        Args:
            nrows(int): the number of rows.
            ncols(int): the number of columns.
            row_nnzs(array-like):
                the number of nonzero entries for each row, or None for an empty matrix.
            values(bool):
                whether it has values or only structure.
        """
        from .constructors import create_empty, create_from_sizes
        if row_nnzs is not None:
            assert len(row_nnzs) == nrows
            if nrows > 0 and np.max(row_nnzs) <= INTC.max:
                row_nnzs = np.require(row_nnzs, np.intc)
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

    if _csr_base is structref.StructRefProxy:
        nrows = property(_struct.get_nrows)
        ncols = property(_struct.get_ncols)
        nnz = property(_struct.get_nnz)
        rowptrs = property(_struct.get_rowptrs)
        colinds = property(_struct.get_colinds)

    @property
    def values(self):
        if NUMBA_ENABLED:
            return _struct.get_values(self)
        else:
            return self._values

    @values.setter
    def values(self, vs: np.ndarray):
        if vs is None:
            new = CSR(self.nrows, self.ncols, self.nnz, self.rowptrs, self.colinds, None)
        else:
            if len(vs) < self.nnz:
                raise ValueError('value array too small')
            elif len(vs) > self.nnz:
                vs = vs[:self.nnz]

            new = CSR(self.nrows, self.ncols, self.nnz, self.rowptrs, self.colinds, vs)

        if NUMBA_ENABLED:
            # replace our internals
            self._type = new._type
            self._meminfo = new._meminfo
        else:
            self._values = new._values

    def _required_values(self):
        """
        Get the value array, returning an array of 1s if it is not present.
        """
        vs = self.values
        if vs is None:
            return np.ones(self.nnz)
        else:
            return vs

    def _e_value(self, i):
        vs = self.values
        if vs is not None:
            return vs[i]
        else:
            return 1.0

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

        .. note:: This method is not available from Numba.

        Args:
            begin(int): the first row index to include.
            end(int): one past the last row to include.

        Returns:
            CSR: the matrix only containing a subset of the rows.  It shares storage
                with the original matrix to the extent possible.
        """
        from .structure import subset_rows
        return subset_rows(self, begin, end)

    def rowinds(self) -> np.ndarray:
        """
        Get the row indices from this array.  Combined with :py:attr:`colinds` and
        :py:attr:`values`, this can form a COO-format sparse matrix.
        """
        return _rows.all_indices(self)

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
        return _rows.array(self, row)

    def row_extent(self, row):
        """
        Get the extent of a row in the underlying column index and value arrays.

        Args:
            row(int): the row index.

        Returns:
            tuple: ``(s, e)``, where the row occupies positions :math:`[s, e)` in the
            CSR data.
        """
        return _rows.extent(self, row)

    def row_cs(self, row):
        """
        Get the column indcies for the stored values of a row.
        """
        return _rows.cs(self, row)

    def row_vs(self, row):
        """
        Get the stored values of a row.  If only the matrix structure is stored, this
        returns a vector of 1s.
        """
        return _rows.vs(self, row)

    def row_nnzs(self):
        """
        Get a vector of the number of nonzero entries in each row.

        .. note:: This method is not available from Numba.

        Returns:
            numpy.ndarray: the number of nonzero entries in each row.
        """
        return np.diff(self.rowptrs)

    def normalize_rows(self, normalization):
        """
        Normalize the rows of the matrix.

        .. note:: The normalization *ignores* missing values instead of treating
                  them as 0.

        .. note:: This method is not available from Numba.

        Args:
            normalization(str):
                The normalization to perform. Can be one of:

                * ``'center'`` - center rows about the mean
                * ``'unit'`` - convert rows to a unit vector

        Returns:
            numpy.ndarray:
                The normalization values for each row.
        """
        from . import transform
        if normalization == 'center':
            return transform.center_rows(self)
        elif normalization == 'unit':
            return transform.unit_rows(self)
        else:
            raise ValueError('unknown normalization: ' + normalization)

    def transpose(self, include_values=True):
        """
        Transpose a CSR matrix.

        .. note::
            In Numba, this method takes no paramters.
            Call :py:meth:`transpose_structure` for a structure-only transpose.

        Args:
            include_values(bool): whether to include the values in the transpose.

        Returns:
            CSR: the transpose of this matrix (or, equivalently, this matrix in CSC format).
        """
        from .structure import transpose
        return transpose(self, include_values)

    def transpose_structure(self):
        """
        Tranpose the structure of a CSR matrix.  The resulting matrix has no values.
        """
        return self.transpose(False)

    def filter_nnzs(self, filt):
        """
        Filter the values along the full NNZ axis.

        .. note:: This method is not available from Numba.

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

    def multiply(self, other, transpose=False):
        """
        Multiply this matrix by another.

        .. note:: In Numba, ``transpose`` is a mandatory positional argument.  Numba users
                  may wish to directly use the kernel API.

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

        .. warning:: This method is deprecated.

        .. note:: This method is not available from Numba.
        """
        warnings.warn('drop_values is deprecated', DeprecationWarning)
        self.values = None

    def fill_values(self, value):
        """
        Fill the values of this CSR with the specified value.  If the CSR is
        structure-only, a value array is added.  This is an **in-place** operation.

        .. warning:: This method is deprecated.

        .. note:: This method is not available from Numba.
        """
        if self.values is not None:
            self.values[:] = value
        else:
            self.values = np.full(self.nnz, value, dtype='float64')

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
        args = (self.nrows, self.ncols, self.nnz, self.rowptrs, self.colinds, self.values, False)
        return (CSR, args)
