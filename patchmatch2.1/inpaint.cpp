#include <cmath>

#include "nn.h"
#include "lua_inpaint.h"
#include "allegro_emu.h"
#include "inpaint.h"

class BITMAP;
class Params;
class RegionMasks;
class Pyramid;


void free_pyramid(Pyramid *pyramid);
void build_pyramid(Pyramid * pyramid, Params *p);

class Pyramid
{
public:
	BITMAP *image_initial; // initial is not copied during construction.
	                       // Avoiding copying or moving (for laziness)
	BITMAP *mask_initial;

	BITMAP **images_pyramid;
	BITMAP **masks_pyramid;

	int max_pyramid_level;

	Pyramid(int max_pyramid_level_=0): max_pyramid_level(max_pyramid_level) {}
};


BITMAP *inpaint(Params *p, BITMAP *a, BITMAP *mask)
{

	Pyramid pyramid;

	// Init the pyramid
	pyramid.image_initial = a;
	pyramid.mask_initial = mask;

	build_pyramid(&pyramid, p);

	free_pyramid(&pyramid);

	return NULL;
}


void build_pyramid(Pyramid * pyramid, Params *p)
{

	BITMAP *image = pyramid->image_initial;
	int max_possible_levels = (int)log(min(image->h, image->w)/p->patch_w);

	if (pyramid->max_pyramid_level != 0 && max_possible_levels > pyramid->max_pyramid_level)
		max_possible_levels = pyramid->max_pyramid_level;
	else
		pyramid->max_pyramid_level = max_possible_levels;

	pyramid->images_pyramid = (BITMAP**)malloc(sizeof(BITMAP*)*pyramid->max_pyramid_level);
	pyramid->masks_pyramid = (BITMAP**)malloc(sizeof(BITMAP*)*pyramid->max_pyramid_level);

	for (int i=0 ; i < pyramid->max_pyramid_level ; i++) {
		pyramid->images_pyramid[i] = downscale_image(pyramid->image_initial);
		pyramid->masks_pyramid[i] = downscale_image(pyramid->mask_initial);
	}
}

void add_level_to_pyramid(Pyramid *pyramid, Params *p, BITMAP *image_source, BITMAP *mask_source, int level)
{
	BITMAP *image = (BITMAP*)malloc(sizeof(BITMAP));
	pyramid->images_pyramid[level] = image;


	// image->data = (unsigned char*)malloc(sizeof(unsigend char)*

}


void free_pyramid(Pyramid *pyramid)
{
	// TODO: complete
}
