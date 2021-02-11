Compressed Sparse Matrices
==========================

.. py:module:: csr

The :py:class:`CSR` class is the entry point for pure Python code to work with the
CSR package.

.. autoclass:: CSR

Constructing Matrices
---------------------

In addition to the CSR constructor, there are several utility methods for constructing
sparse matrices.

.. automethod:: CSR.from_coo
.. automethod:: CSR.empty

Constructing from Numba
~~~~~~~~~~~~~~~~~~~~~~~

Numba does not provide access to CSR's class methods; instead, use the creation
functions (these also work from pure Python):

.. autofunction:: create
.. autofunction:: create_novalues
.. autofunction:: create_empty
.. autofunction:: create_from_sizes

Accessing Rows
--------------

The CSR data itself is exposed through attributes.  There are also several methods to
extract row data in a more convenient form.

.. automethod:: CSR.row_extent
.. automethod:: CSR.row_cs
.. automethod:: CSR.row_vs
.. automethod:: CSR.row

Transforming and Manipulating Matrices
--------------------------------------

.. automethod:: CSR.copy
.. automethod:: CSR.subset_rows
.. automethod:: CSR.filter_nnzs
.. automethod:: CSR.transpose
.. automethod:: CSR.normalize_rows
.. automethod:: CSR.drop_values
.. automethod:: CSR.fill_values

Arithmetic
----------

CSRs do not yet support the full suite of SciPy/NumPy matrix operations, but they do
support multiplications:

.. automethod:: CSR.mult_vec
.. automethod:: CSR.multiply

SciPy Integration
-----------------

CSR matrices can be converted to and from SciPy sparse matrices (in any layout):

.. automethod:: CSR.from_scipy
.. automethod:: CSR.to_scipy

