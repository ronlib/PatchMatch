struct BITMAP;

int nn16_patch_dist(int *adata, BITMAP *b, int bx, int by, int maxval, Params *p);
int nn16_patch_dist_ab(BITMAP *a, int ax, int ay, BITMAP *b, int bx, int by, int maxval, Params *p);
BITMAP* downscale_image(BITMAP *image);
BITMAP *scale_image(BITMAP *image, int hs, int ws);
void save_bitmap(BITMAP *bmp, const char *filename);
