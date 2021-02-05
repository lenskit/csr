Native Representation
=====================

The :py:class:`csr.CSR` class provides the primary object-oriented interface to
the sparse matrix capabilities for use from Python, but is not directly usable
in functions written for use with Numba's nopython mode.  Therefore, the Python
class is written as a thin wrapper around a Numba jitclass, :py:class:`csr._CSR`,
that can be efficiently used from compiled code.

This class only provides storage, not methods.  Native functions corresponding
to most of the CSR operations are provided by :py:mod:`csr.native_ops` and the
kernels_.

.. _kernels: kernels.html

.. autoclass:: csr._CSR

Native Operations
-----------------

.. automodule:: csr.native_ops
    :members:
