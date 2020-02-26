import os
import glob
from PyInstaller.compat import is_win
from PyInstaller.utils.hooks import get_module_file_attribute

binaries = []
binaries.append((os.path.join(os.path.dirname(
get_module_file_attribute('sklearn')), ".libs", "vcomp140.dll"), "sklearn/.libs/"))

hiddenimports = ['sklearn.utils._cython_blas',
                 'sklearn.neighbors.typedefs',
                 'sklearn.neighbors.quad_tree',
                 'sklearn.tree',
                 'sklearn.tree._utils']
