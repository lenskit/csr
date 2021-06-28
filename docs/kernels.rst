Kernels
=======

.. py:module:: csr.kernels

More sophisticated CSR operations, such as multiplications, are provided by a *kernel*.
Numba supports multiple kernels; at import time, a default kernel is loaded and is
available as `csr.kernel`.  Numba-compiled functions using the CSR kernel should
access it through this module.

There are currently three kernels available:

numba
    This kernel implements the operations directly with Numba-compiled functions.

mkl
    This kernel uses the Intel Math Kernel Library (MKL) to implement the operations.

scipy
    This kernel is not intended for general use, and only exists to make testing and
    benchmarking easier.  It uses SciPy's sparse matrices to implement matrix math,
    and **cannot be used from Numba**.

The default kernel is automatically selected when ``csr`` is imported.  It is selected
as follows:

1.  If the environment variable ``CSR_KERNEL`` is set, its value is used as the kernel
    name.
2.  If the MKL kernel is compiled and MKL is available, it is used.
3.  Otherwise, the Numba kernel is used.

The PyPI packages do **not** include the MKL kernel.  CSR only supports MKL when used
with Conda; the conda-forge packages for Intel platforms include the MKL kernel.

More kernels may be added in the future; for example, it may be useful to add a CUDA kernel,
or if an optimized sparse matrix package for 64-bit ARM becomes available.

There are two ways to access the kernel.  One is to import :py:mod:`csr.kernel` and use
the functions directly; this uses the kernel selected at import time and cannot be
dynamically changed.  Numba-optimized functions must use this access method; pure Python
code generally doesn't need to access kernels directly, but may use either this method
or dynamic selection (described in the next section).

Dynamic Kernel Selection
------------------------

The kernel used by the Python APIs can be changed at runtime as well.  The Python implementations of
the various methods provided by :py:class:`csr.CSR` use this API to access the active kernel, so
changing the kernel with :py:func:`set_kernel` will change the kernel for pure Python code using
``CSR`` and its methods, but will not change the code used for Numba-based code.  Dynamic kernel
selection is primarily useful for testing and benchmarking.

.. note::
    This does **not** change the kernel exposed as ``csr.kernel``, which is used by
    Numba-compiled client functions.

.. autofunction:: get_kernel
.. autofunction:: set_kernel
.. autofunction:: use_kernel

Kernel Interface
----------------

.. py:module:: csr.kernel

The :py:mod:`csr.kernel` module exposes the kernel interface.  These same functions are
available on any kernel, including those returned by :py:func:`csr.kernels.get_kernel`.

Handles
~~~~~~~

The kernel interface is built on opaque *handles*: a :py:class:`csr._CSR` needs to be
converted to a handle with :py:func:`to_handle`, and subsequent operations use that
handle to access the matrix.  Handles must be *explicitly* released, or they will
generally leak memory.

.. autofunction:: to_handle
.. autofunction:: from_handle
.. autofunction:: release_handle

Manipulation
~~~~~~~~~~~~

.. autofunction:: order_columns

Multiplication
~~~~~~~~~~~~~~

.. autofunction:: mult_vec
.. autofunction:: mult_ab
.. autofunction:: mult_abt

Additional Requirements
~~~~~~~~~~~~~~~~~~~~~~~

There are additional requirements for kernel implementations:

.. py:attribute:: max_nnz
    :type: int

    This attribute stores the maximum number of non-zero entries supported by this kernel. It is not
    exposed on the kernel module, but is exposed on dynamically-obtained kernels, and must be
    provided by any kernel implementation.  It is used by :py:class:`csr._CSR` to automatically
    handle matrices that are too large for a particular kernel when possible.
