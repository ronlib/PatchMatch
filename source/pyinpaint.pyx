# cimport pywrapped_inpaint

cimport c_pyinpaint
from libc.stdlib cimport free
import numpy as np
cimport numpy as np

np.import_array()


cdef extern from "<stdio.h>" nogil:
    int printf   (const char *template, ...)


cdef extern from "pywrapped_inpaint.c":
    void mem_alloc(int ** out_buf, int* allocated_int_length)

def pyinpaint(img, nchannels, mask, H, W):
    cdef unsigned char * output_buf
    pyshape = img.shape
    c_pyinpaint.pyinpaint(img.tobytes(), nchannels, mask.tobytes(), H, W, &output_buf)
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> H*W*nchannels
    nparr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UBYTE, output_buf)
    nparr = nparr.reshape(pyshape)
    return nparr



def pymem_alloc():
    cdef int* carr
    cdef int carr_len
    cdef np.npy_intp shape[1]
    mem_alloc(&carr, &carr_len)
    print (carr_len)
    printf ("%p\n", carr)
    shape[0] = <np.npy_intp> carr_len
    nparr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, carr)
    # return nparr
    return nparr

cdef public void printer():
    print ('bla in printer()');
