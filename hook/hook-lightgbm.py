import os
import glob
from PyInstaller.compat import is_win
from PyInstaller.utils.hooks import get_module_file_attribute

binaries = []
binaries.append((os.path.join(os.path.dirname(
get_module_file_attribute('lightgbm')), "lib_lightgbm.dll"), "lightgbm"))

