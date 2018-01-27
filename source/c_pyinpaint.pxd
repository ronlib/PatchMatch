cdef extern from "pywrapped_inpaint.c":
    int pyinpaint(unsigned char* img_data, int nchannels, unsigned char* in_mask, int H, int W, unsigned char ** out_buf)
