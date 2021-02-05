"""
Python API for CSR matrices.
"""

from numba.core.types.abstract import InitialValue
import numpy as np
import scipy.sparse as sps

from csr.kernels import get_kernel, releasing
from csr import native_ops as _ops
from csr.layout import EMPTY_VALUES, _CSR


def _csr_delegate(name):
    def func(self):
        return getattr(self.R, name)

    return property(func)


class CSR:
    """
    Simple compressed sparse row matrix.  This is like :py:class:`scipy.sparse.csr_matrix`, with
    a couple of useful differences:

    * The value array is optional, for cases in which only the matrix structure is required.
    * The value array, if present, is always double-precision.

    You generally don't want to create this class yourself with the constructor.  Instead, use one
    of its class methods.

    It is backed by separate storage type (py:class:`csr._CSR`) that can be passed around through
    Numba-compiled functions, and nopython compiled equivalents of many of its methods
    are available as functions in the :py:mod:`csr.native_ops` module that take the
    underlying tuple (accessible by :py:attr:`R`) as a parameter.

    If you need to pass an instance off to a Numba-compiled function, use :py:attr:`R`::

        _some_numba_fun(csr.R)

    Attributes:
        R(_CSR): the named tuple containing the actual matrix data
        nrows(int): the number of rows.
        ncols(int): the number of columns.
        nnz(int): the number of entries.
        rowptrs(numpy.ndarray): the row pointers.
        colinds(numpy.ndarray): the column indices.
        values(numpy.ndarray or None): the values
    """
    __slots__ = ['R']

    def __init__(self, nrows=None, ncols=None, nnz=None, ptrs=None, inds=None, vals=None, R=None):
        if R is not None:
            assert R.colinds.dtype == np.int32
            self.R = R
        else:
            assert inds.dtype == np.int32
            if vals is None:
                vals = EMPTY_VALUES
            self.R = _CSR(nrows, ncols, nnz, ptrs, inds, vals)

    @classmethod
    def empty(cls, shape, row_nnzs, *, rpdtype=np.intc):
        """
        Create an empty CSR matrix.

        Args:
            shape(tuple): the array shape (rows,cols)
            row_nnzs(array-like): the number of nonzero entries for each row
        """
        nrows, ncols = shape
        assert len(row_nnzs) == nrows

        nnz = np.sum(row_nnzs, dtype=np.int64)
        rowptrs = np.zeros(nrows + 1, dtype=rpdtype)
        rowptrs[1:] = np.cumsum(row_nnzs, dtype=rpdtype)
        colinds = np.full(nnz, -1, dtype=np.intc)
        values = np.full(nnz, np.nan)

        return cls(nrows, ncols, nnz, rowptrs, colinds, values)

    @classmethod
    def from_coo(cls, rows, cols, vals, shape=None, rpdtype=np.intc):
        """
        Create a CSR matrix from data in COO format.

        Args:
            rows(array-like): the row indices.
            cols(array-like): the column indices.
            vals(array-like): the data values; can be ``None``.
            shape(tuple): the array shape, or ``None`` to infer from row & column indices.
        """
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

        rowptrs = np.zeros(nrows + 1, dtype=rpdtype)
        align = np.full(nnz, -1, dtype=rpdtype)

        _ops._csr_align(rows, nrows, rowptrs, align)

        cols = cols[align].copy()
        vals = vals[align].copy() if vals is not None else None

        return cls(nrows, ncols, nnz, rowptrs, cols, vals)

    @classmethod
    def from_scipy(cls, mat, copy=True):
        """
        Convert a scipy sparse matrix to an internal CSR.

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

    nrows = _csr_delegate('nrows')
    ncols = _csr_delegate('ncols')
    nnz = _csr_delegate('nnz')
    rowptrs = _csr_delegate('rowptrs')
    colinds = _csr_delegate('colinds')

    @property
    def values(self):
        return self.R.values if self.R.has_values else None

    def subset_rows(self, begin, end):
        """
        Subset the rows in this matrix.
        """
        return CSR(N=_ops._subset_rows(self.R, begin, end))

    def rowinds(self) -> np.ndarray:
        """
        Get the row indices from this array.  Combined with :py:attr:`colinds` and
        :py:attr:`values`, this can form a COO-format sparse matrix.
        """
        # we have to upgrade to Numba for this one - too slow in Python
        return _ops.rowinds(self.R)

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
        return _ops.row(self.R, row)

    def row_extent(self, row):
        """
        Get the extent of a row in the underlying column index and value arrays.

        Args:
            row(int): the row index.

        Returns:
            tuple: ``(s, e)``, where the row occupies positions :math:`[s, e)` in the
            CSR data.
        """
        return _ops.row_extent(self.R, row)

    def row_cs(self, row):
        """
        Get the column indcies for the stored values of a row.
        """
        return _ops.row_cs(self.R, row)

    def row_vs(self, row):
        """
        Get the stored values of a row.  If only the matrix structure is stored, this
        returns a vector of 1s.
        """
        return _ops.row_vs(self.R, row)

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
        if normalization == 'center':
            return _ops.center_rows(self.R)
        elif normalization == 'unit':
            return _ops.unit_rows(self.R)
        else:
            raise ValueError('unknown normalization: ' + normalization)

    def transpose(self, values=True):
        """
        Transpose a CSR matrix.

        .. note:: This method is not available from Numba.

        Args:
            values(bool): whether to include the values in the transpose.

        Returns:
            CSR: the transpose of this matrix (or, equivalently, this matrix in CSC format).
        """

        r2 = _ops.transpose(self.R, values)
        return CSR(R=r2)

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
            rps2[i+1] = rps2[i] + rlen

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
        with releasing(K.to_handle(self.R), K) as a_h:
            with releasing(K.to_handle(other.R), K) as b_h:
                if transpose:
                    c_h = K.mult_abt(a_h, b_h)
                else:
                    c_h = K.mult_ab(a_h, b_h)
                with releasing(c_h, K):
                    crepr = K.from_handle(c_h)

        return CSR(R=crepr)

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
        with releasing(K.to_handle(self.R), K) as h:
            return K.mult_vec(h, v)

    def drop_values(self):
        """
        Remove the value array from this CSR.
        """
        self.R.values = EMPTY_VALUES
        self.R.has_values = False

    def fill_values(self, value):
        """
        Fill the values of this CSR with the specified value.  If the CSR is
        structure-only, a value array is added.
        """
        self.R.values = np.full(self.nnz, value, dtype='float64')
        self.R.has_values = True

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

    def __getstate__(self):
        return dict(shape=(self.nrows, self.ncols), nnz=self.nnz,
                    rowptrs=self.rowptrs, colinds=self.colinds, values=self.values)

    def __setstate__(self, state):
        nrows, ncols = state['shape']
        nnz = state['nnz']
        rps = state['rowptrs']
        cis = state['colinds']
        vs = state['values']
        if vs is None:
            vs = EMPTY_VALUES
        self.R = _CSR(nrows, ncols, nnz, rps, cis, vs)
