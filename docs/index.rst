CSR
===

The CSR library provides a sparse matrix class (in compressed sparse row format)
along with various matrix operations.  Its capabilities are a subset of those
provided by :py:class:`scipy.sparse.csr_matrix`, with both the matrix type itself
and most of the operations accessible from Numba's nopython mode.

The :py:class:`csr.CSR` class is the main entry point for using this package.

CSR is not currently suitable for use as a general-purpose sparse matrix package.
It is quite good at representing sparse matrices in a form suitable for custom
computations with Numba, and when the Intel MKL is available its Sparse BLAS is
used to accelerate several operations.

Contents
--------

.. toctree::
   :maxdepth: 1

   csr
   kernels

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Acknowledgements
================

This material is based upon work supported by the National Science Foundation
under Grant No. IIS 17-51278. Any opinions, findings, and conclusions or
recommendations expressed in this material are those of the author(s) and do not
necessarily reflect the views of the National Science Foundation.
