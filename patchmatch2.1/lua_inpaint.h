struct BITMAP;
struct Params;

BITMAP* downscale_image(BITMAP *image);
BITMAP *scale_image(BITMAP *image, int hs, int ws);
void save_bitmap(BITMAP *bmp, const char *filename);
int nn_patch2vec(BITMAP *a, int ax, int ay, Params *p, float *ret_arr);

extern struct lua_State * g_L;
