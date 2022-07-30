import os
import sys
import warnings
from shutil import rmtree
from pathlib import Path
from cffi import FFI

from invoke import task
try:
    from lkbuild.tasks import *  # NOQA: F403, F401
except ImportError:
    warnings.warn('lkbuild tasks not available')


@task
def clean(c):
    "Clean up build products"

    print('removing build dir', file=sys.stderr)
    rmtree('build')
    print('removing dist dir', file=sys.stderr)
    rmtree('dist')

    pkg_dir = Path(__file__).parent / 'csr' / 'kernels' / 'mkl'
    for f in pkg_dir.glob('_mkl_ops.*'):
        print('removing', f, file=sys.stderr)
        f.unlink(True)

@task
def build_mkl(c, trace=False):
    "Build the Intel MKL helper module."

    pkg_dir = Path(__file__).parent / 'csr' / 'kernels' / 'mkl'
    src_path = pkg_dir / 'mkl_ops.c'
    hdr_path = pkg_dir / 'mkl_ops.h'

    if 'CONDA_PREFIX' in os.environ:
        base = Path(os.environ['CONDA_PREFIX'])
    else:
        warnings.warn('No CONDA_PREFIX set, trying to buil MKL with sys.prefix')
        base = Path(sys.prefix)
    i_dirs = [os.fspath(pkg_dir)]
    l_dirs = []
    if os.name == 'nt':
        lib = base / 'Library'
        i_dirs.append(os.fspath(lib / 'include'))
        l_dirs.append(os.fspath(lib / 'lib'))
    else:
        i_dirs.append(os.fspath(base / 'include'))
        l_dirs.append(os.fspath(base / 'lib'))

    ffibuilder = FFI()
    ffibuilder.cdef(hdr_path.read_text().replace('EXPORT ', ''))

    defines = []
    if trace:
        defines.append(('LK_TRACE', None))
    ffibuilder.set_source("csr.kernels.mkl._mkl_ops", src_path.read_text(),
                          include_dirs=i_dirs, define_macros=defines,
                          libraries=['mkl_rt'], library_dirs=l_dirs)
    ffibuilder.compile(verbose=True)
