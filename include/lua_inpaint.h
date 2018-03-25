#ifndef __LUA_INPAINT_H
#define __LUA_INPAINT_H

#ifdef DPNN_PATCH_DISTANCE
#include "structdef.h"

int distanceDPNNMaskedImage(MaskedImage_P source,int xs,int ys, MaskedImage_P target,int xt,int yt, int S);

#endif

#endif
