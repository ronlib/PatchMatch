from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import subprocess

proc_libs = subprocess.check_output("pkg-config --libs opencv".split())
libs = [lib.lstrip('-l').lstrip('L/') for lib in str(proc_libs, "utf-8").split()[1:]]
proc_incs = subprocess.check_output("pkg-config --cflags opencv".split())

extensions = [
    Extension("pyinpaint", ["source/pyinpaint.pyx"],
              include_dirs=["include", "/usr/include/opencv"],
              libraries=libs)
    ]

setup(
    name = "My inpaint app",
    # ext_modules = cythonize("source/*.pyx", include_dirs = ["./include/"])
    ext_modules = cythonize(extensions)
)
