# cimport pywrapped_inpaint

cdef extern from "pywrapped_inpaint.c":
    int pyinpaint(unsigned char* img, int nchannels, unsigned char* in_mask, int H, int W)

def pyinpaint2(img, nchannels, mask, H, W):
    pyinpaint(img, nchannels, mask, H, W)
