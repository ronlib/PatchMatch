#ifndef _LUA_INPAINT_H
#define _LUA_INPAINT_H
#include <mutex>
struct BITMAP;
class Params;

BITMAP* downscale_image(BITMAP *image);
BITMAP *scale_image(BITMAP *image, int hs, int ws);
void save_bitmap(BITMAP *bmp, const char *filename);
int nn_patch2vec(BITMAP *a, int ax, int ay, Params *p, float *ret_arr);
void init_p2v(BITMAP *im);
void zero_p2v(BITMAP *im);

extern struct lua_State * g_L;

typedef struct Counters {
  Counters();
  unsigned long int n_patch_comp; // number of patch comparisons requested
  unsigned long int n_nn_references; // number of neural network references requested
  unsigned long int n_cache_nn; // number of neural network cached results returned
  std::mutex m;
} Counters;

extern Counters g_counters;

#endif //_LUA_INPAINT_H
