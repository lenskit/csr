# CSR Matrices for Python

[![codecov](https://codecov.io/gh/lenskit/csr/branch/main/graph/badge.svg?token=6UQ4O8FCKK)](https://codecov.io/gh/lenskit/csr)

> [!NOTE]
> Development of this package is suspended as of June 2024, as LensKit 2024.x will
> not use it. Small maintenance releases to support LensKit 0.14 may be maintained.
>
> If someone is interested in taking this package over for other purposes, please
> contact @mdekstrand.

This package provides an implementation of sparse matrices in compressed sparse
row format for Python. Routines are implemented with Numba, and both the CSR
data structure and most related matrix and vector operations can be used from
Numba's nopython mode.

Right now the feature set is very limited --- don't expect this to be a drop-in
replacement for SciPy's sparse matrices.  Some features we expect to develop as
people take an interest in the package and contribute updates, but we also hope
to keep a relatively tight scope.  This package aims to provide efficient support
for compressed sparse row matrices, with some routines that can also treat them
as CSC matrices.
