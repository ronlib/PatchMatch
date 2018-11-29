#ifndef _inpaint_h
#define _inpaint_h


class BITMAP;
class Params;

BITMAP *inpaint(Params *p, BITMAP *a, BITMAP *mask, bool add_completion=false);


#endif
