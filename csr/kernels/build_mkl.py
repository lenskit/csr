import os
import sys
from pathlib import Path
from cffi import FFI

pkg_dir = Path(__file__).parent / 'mkl'
src_path = pkg_dir / 'mkl_ops.c'
hdr_path = pkg_dir / 'mkl_ops.h'

conda = Path(os.environ['CONDA_PREFIX'])
i_dirs = [os.fspath(pkg_dir)]
l_dirs = []
if os.name == 'nt':
    lib = conda / 'Library'
    i_dirs.append(os.fspath(lib / 'include'))
    l_dirs.append(os.fspath(lib / 'lib'))
else:
    i_dirs.append(os.fspath(conda / 'include'))
    l_dirs.append(os.fspath(conda / 'lib'))

ffibuilder = FFI()
ffibuilder.cdef(hdr_path.read_text().replace('EXPORT ', ''))

if __name__ == '__main__':
    defines = []
    if '--trace' in sys.argv:
        defines.append(('LK_TRACE', None))
    ffibuilder.set_source("csr.kernels.mkl._mkl_ops", src_path.read_text(),
                          include_dirs=i_dirs, define_macros=defines,
                          libraries=['mkl_rt'], library_dirs=l_dirs)
    ffibuilder.compile(verbose=True)
