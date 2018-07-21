struct BITMAP;

BITMAP* downscale_image(BITMAP *image);
BITMAP *scale_image(BITMAP *image, int hs, int ws);
void save_bitmap(BITMAP *bmp, const char *filename);

extern struct lua_State * g_L;
