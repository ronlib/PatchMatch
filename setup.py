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
    extra_link_args = None

elif os.name == "nt":
    opencv_installtion_path = r"c:\users\john\Downloads\opencv"
    include_names = ['highgui','gpu','ts','calib3d','features2d','stitching','ocl','superres','contrib','core','video','videostab','legacy','objdetect','nonfree','flann','photo','imgproc','ml',]
    incs = [os.path.join(opencv_installtion_path, 'sources', 'modules', i, 'include') for i in include_names]
    incs.append(os.path.join(opencv_installtion_path, 'sources', 'include', 'opencv'))
    incs.append(os.path.join(opencv_installtion_path, 'sources', 'include'))
    incs.append('include')
    incs.append(np.get_include())
    libs = ['opencv_core2413', 'opencv_imgproc2413', 'zlib',]
    extra_link_args = ["/LIBPATH:"+os.path.join(opencv_installtion_path, 'build', 'x86', 'vc14', 'staticlib')]


extensions = [
    Extension("pyinpaint", ["source/pyinpaint.pyx", "source/inpaint.c", "source/maskedimage.c", "source/nearestneighborfield.c"],
        include_dirs=incs,
        libraries=libs,
        extra_link_args=extra_link_args,
            ),
    ]

setup(
    name = "Inpaint extension",
    ext_modules = cythonize(extensions, gdb_debug=True)
)
