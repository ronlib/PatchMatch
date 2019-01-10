struct BITMAP;
class Params;

BITMAP* downscale_image(BITMAP *image);
BITMAP *scale_image(BITMAP *image, int hs, int ws);
void save_bitmap(BITMAP *bmp, const char *filename);
int nn_patch2vec(BITMAP *a, int ax, int ay, Params *p, float *ret_arr);
void init_p2v(BITMAP *im);
void zero_p2v(BITMAP *im);

extern struct lua_State * g_L;
