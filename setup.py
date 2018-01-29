from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import subprocess
import os
import numpy as np

if os.name == "posix":
    proc_libs = subprocess.check_output("pkg-config --libs opencv".split())
    libs = [lib.lstrip('-l').lstrip('L/') for lib in str(proc_libs, "utf-8").split()[1:]]
    incs = ["include", "/usr/include/opencv",]

    includes = ["include", "/usr/include/opencv",]
elif os.name == "nt":
    opencv_installtion_path = r"c:\users\john\Downloads\opencv"
    # opencv_relative_includes = [["sources", "include", "opencv"], ["sources", "include",],["sources", "modules", "core", "include"]]
    # incs = [os.path.join(opencv_installtion_path, *rel_path) for rel_path in opencv_relative_includes]
    include_names = ['highgui','gpu','ts','calib3d','features2d','stitching','ocl','superres','contrib','core','video','videostab','legacy','objdetect','nonfree','flann','photo','imgproc','ml',]
    incs = [os.path.join(opencv_installtion_path, 'sources', 'modules', i, 'include') for i in include_names]
    incs.append(os.path.join(opencv_installtion_path, 'sources', 'include', 'opencv'))
    incs.append(os.path.join(opencv_installtion_path, 'sources', 'include'))
    incs.append('include')
    incs.append(np.get_include())

    libs = []


extensions = [
    Extension("pyinpaint", ["source/pyinpaint.pyx", "source/inpaint.c", "source/maskedimage.c", "source/nearestneighborfield.c"],
        include_dirs=incs,
        libraries=libs),
    ]

setup(
    name = "My inpaint app",
    # ext_modules = cythonize("source/*.pyx", include_dirs = ["./include/"])
    ext_modules = cythonize(extensions, gdb_debug=True)
)
