"""
Python API for CSR matrices.
"""

import warnings
import logging
import numpy as np
import scipy.sparse as sps

from numba import config
from numba.experimental import structref

from csr.kernels import get_kernel, releasing
from . import _struct, _rows

INTC = np.iinfo(np.intc)
_log = logging.getLogger(__name__)

# ugly hack for a bug on Numba < 0.53
if config.DISABLE_JIT:
    class _csr_base:
        def __init__(self, nrows, ncols, nnz, ptrs, inds, vals, _cast=True):
            self.nrows = nrows
            self.ncols = ncols
            self.nnz = nnz
            if _cast and np.max(ptrs, initial=0) <= INTC.max:
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
    of its class or static methods.  If you do use the constructor, be advised that the class may
    reuse the arrays that you pass, but does not guarantee that they will be used.

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

    def __new__(cls, nrows, ncols, nnz, rps, cis, vs, _cast=True):
        assert nrows >= 0
        assert nrows <= INTC.max
        assert ncols >= 0
        assert ncols <= INTC.max
        assert nnz >= 0
        nrows = np.intc(nrows)
        ncols = np.intc(ncols)

        if _cast:
            cis = np.require(cis, np.intc, 'C')
            if nnz <= INTC.max:
                rps = np.require(rps, np.intc, 'C')
            else:
                rps = np.require(rps, np.int64, 'C')
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
            values(bool, str, or numpy.dtype):
                whether it has values or only structure; can be a NumPy data type to
                specify a type other than `f8`.
        """
        from .constructors import create_empty
        assert nrows >= 0
        assert ncols >= 0
        if row_nnzs is not None:
            assert len(row_nnzs) == nrows
            nnz = np.sum(row_nnzs, dtype=np.int64)
            assert nnz >= 0
            rp_dtype = np.intc if nnz <= INTC.max else np.int64
            rps = np.zeros(nrows + 1, dtype=rp_dtype)
            np.cumsum(row_nnzs, dtype=rp_dtype, out=rps[1:])
            cis = np.zeros(nnz, dtype=np.int32)

            if values is True:
                vs = np.zeros(nnz)
            elif values:
                vs = np.zeros(nnz, dtype=values)
            else:
                vs = None

            return cls(nrows, ncols, nnz, rps, cis, vs)
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

        assert np.min(rows, initial=0) >= 0
        assert np.min(cols, initial=0) >= 0

        if shape is not None:
            nrows, ncols = shape
            # if rows/cols is 0, that's fine; max must be zero
            assert np.max(rows, initial=0) < max(nrows, 1)
            assert np.max(cols, initial=0) < max(ncols, 1)
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
        """
        Get the value of a particular element, returning 1 if values is undefined.
        """
        vs = self.values
        if vs is not None:
            return vs[i]
        else:
            return 1.0

    def _normalize(self, val_dtype=np.float64, ptr_dtype=None):
        """
        Normalize the matrix into a predictable structure and type.  It avoids copying
        if possible.

        .. note:: This method is not available from Numba.

        Args:
            val_dtype(np.dtype or None or boolean):
                The value data type.  If ``False``, drop the value array.  If ``None``,
                leave unchanged.
            ptr_dtype(np.dtype or None):
                The row pointer data type.  If ``None``, leave rows untransformed.
        Returns:
            CSR: the transformed CSR matrix.
        """

        if ptr_dtype:
            info = np.iinfo(ptr_dtype)
            if self.nnz > info.max:
                raise ValueError(f'type {ptr_dtype} cannot address {self.nnz} entries')
            rps = np.require(self.rowptrs, ptr_dtype)
        else:
            rps = self.rowptrs

        if val_dtype:
            if self.values is None:
                vs = np.ones(self.nnz, val_dtype)
            else:
                vs = np.require(self.values, val_dtype)
        elif val_dtype is False:
            vs = None
        else:
            vs = self.values

        return CSR(self.nrows, self.ncols, self.nnz, rps, self.colinds, vs, _cast=False)

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

    def pick_rows(self, rows, *, include_values=True):
        """
        Pick rows from this matrix.  A row may appear more than once.

        .. note:: This method is not available from Numba.

        Args:
            rows(numpy.ndarray): the row indices to select.
            include_values(bool): whether to include values if present

        Returns:
            CSR: the matrix containing the specified rows.
        """
        from .structure import _pick_rows, _pick_rows_nvs
        if include_values and self.values is not None:
            return _pick_rows(self, rows)
        else:
            return _pick_rows_nvs(self, rows)

    def rowinds(self) -> np.ndarray:
        """
        Get the row indices from this array.  Combined with :py:attr:`colinds` and
        :py:attr:`values`, this can form a COO-format sparse matrix.
        """
        return _rows.all_indices(self)

    def row(self, row):
        """
        Return one or more rows of this matrix as a dense ndarray.

        Args:
            row(int or numpy.ndarray): the row index or indices.

        Returns:
            numpy.ndarray:
                the row, with 0s in the place of missing values.  If the CSR only
                stores matrix structure, the returned vector has 1s where the CSR
                records an entry.
        """
        row = np.asarray(row, dtype='i4')
        return _rows.row_array(self, row)

    def row_mask(self, row):
        """
        Return a dense logical array indicating which columns are set in the row
        (or rows).

        Args:
            row(int or numpy.ndarray): the row index or indices.

        Returns:
            numpy.ndarray:
                the row, with ``True`` for columns that are set on this row.  If
                ``row`` is an array or list of length :math:`k`, this is a matrix
                of shape :math:`k \\times \\mathrm{ncols}`.
        """
        row = np.asarray(row, dtype='i4')
        return _rows.row_mask(self, row)

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

        # Helper for handling sharding
        def mul(A, b_h):
            with releasing(K.to_handle(A), K) as a_h:
                if transpose:
                    c_h = K.mult_abt(a_h, b_h)
                else:
                    c_h = K.mult_ab(a_h, b_h)
                with releasing(c_h, K):
                    crepr = K.from_handle(c_h)

            crepr._filter_zeros()
            return crepr

        if self.nnz <= K.max_nnz:
            # Common / fast path - one matrix
            with releasing(K.to_handle(other), K) as b_h:
                return mul(self, b_h)
        else:
            # Too large, let's go sharding
            shards = self._shard_rows(K.max_nnz)
            with releasing(K.to_handle(other), K) as b_h:
                sparts = [mul(s, b_h) for s in shards]
            return CSR._assemble_shards(sparts)

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
        if self.nnz <= K.max_nnz:
            with releasing(K.to_handle(self), K) as h:
                return K.mult_vec(h, v)
        else:
            shards = self._shard_rows(K.max_nnz)
            svs = []
            for s in shards:
                with releasing(K.to_handle(s), K) as h:
                    svs.append(K.mult_vec(h, v))
            return np.concatenate(svs)

    def _filter_zeros(self):
        """
        Filter out the stored zero values in-place.
        """
        if self.values is not None:
            _struct.filter_zeros(self)

    def _shard_rows(self, tgt_nnz):
        """
        Shard a matrix by rows to fit in a target size.
        """
        assert tgt_nnz > 0

        rest = self
        shards = []
        while rest.nnz > tgt_nnz:
            # find the first split point
            split = np.searchsorted(rest.rowptrs, tgt_nnz)
            # if the start of the found row is too large, back up by one
            if rest.rowptrs[split] > tgt_nnz:
                if split <= 1:
                    raise ValueError("row too large to fit in target matrix size")
                split -= 1

            _log.debug('splitting %s at %d (rp@s: %d)', rest, split, rest.rowptrs[split])
            shards.append(rest.subset_rows(0, split))
            rest = rest.subset_rows(split, rest.nrows)

        shards.append(rest)
        return shards

    @classmethod
    def _assemble_shards(cls, shards):
        """
        Reassemble a matrix from sharded rows.
        """
        nrows = sum(s.nrows for s in shards)
        ncols = max(s.ncols for s in shards)
        nnz = sum(s.nnz for s in shards)

        rps = np.zeros(nrows + 1, np.int64)
        rs = 0
        for s in shards:
            off = rps[rs]
            re = rs + s.nrows + 1
            rps[rs:re] = s.rowptrs + off
            rs += s.nrows

        assert rps[nrows] == nnz, f'{rps[nrows]} != {nnz}'

        cis = np.concatenate([s.colinds for s in shards])
        assert len(cis) == nnz
        if shards[0].values is not None:
            vs = np.concatenate([s.values for s in shards])
            assert len(vs) == nnz
        else:
            vs = None

        return cls(nrows, ncols, nnz, rps, cis, vs)

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
        repr += '  dtype={}\n'.format(self.values.dtype if self.values is not None else None)
        repr += '}>'
        return repr

    def __reduce__(self):
        args = (self.nrows, self.ncols, self.nnz, self.rowptrs, self.colinds, self.values, False)
        return (CSR, args)
