#include <cmath>
#include <algorithm>
#include <unistd.h>

#include "nn.h"
#include "lua_inpaint.h"
#include "allegro_emu.h"
#include "inpaint.h"

class BITMAP;
class Params;
class RegionMasks;
class Pyramid;


void free_pyramid(Pyramid *pyramid);
void build_pyramid(Params *p, Pyramid * pyramid, BITMAP *image, BITMAP *mask);
void inpaint_image(Params *p, BITMAP *image, BITMAP *mask, BITMAP *inv_mask);
void copy_unmasked_nnf_regions(Params *p, BITMAP *image, RegionMasks *amask, BITMAP *ann);
int is_point_in_box(int y, int x, Box box, int border);
BITMAP *inverse_mask_bitmap(BITMAP *mask);
BITMAP *threshold_image(BITMAP *image, unsigned char threshold);

/*
  This function goes over the mask, looking for masked pixels which are at most
  min_edge_distance from the right or lower border, and moves it upwards or
  left (or both).

  Returns the original mask

*/
BITMAP *translate_mask(BITMAP *mask, unsigned int translation, unsigned int min_edge_distance);

class Pyramid
{
public:
	BITMAP *image_initial; // initial is not copied during construction.
	                       // Avoiding copying or moving (for laziness)
	BITMAP *mask_initial;

	BITMAP **images_pyramid;
	BITMAP **masks_pyramid;
  BITMAP **inv_masks_pyramid;

	int max_pyramid_level;

	Pyramid(int max_pyramid_level_=0): max_pyramid_level(max_pyramid_level_) {}
};


BITMAP *inpaint(Params *p, BITMAP *a, BITMAP *mask)
{

	Pyramid pyramid;

	// Init the pyramid
	// pyramid.image_initial = a;
	// pyramid.mask_initial = mask;

	build_pyramid(p, &pyramid, a, mask);

  inpaint_image(p, pyramid.images_pyramid[pyramid.max_pyramid_level-1],
                pyramid.masks_pyramid[pyramid.max_pyramid_level-1],
                pyramid.inv_masks_pyramid[pyramid.max_pyramid_level-1]);

  // vote

	free_pyramid(&pyramid);

	return NULL;
}


void build_pyramid(Params *p, Pyramid * pyramid, BITMAP *image, BITMAP *mask)
{

	// BITMAP *image = pyramid->image_initial;
	int max_possible_levels = (int)log2((double)std::min(image->h, image->w)/p->patch_w) + 1;

	if (pyramid->max_pyramid_level != 0 && max_possible_levels > pyramid->max_pyramid_level)
		max_possible_levels = pyramid->max_pyramid_level;
	else
		pyramid->max_pyramid_level = max_possible_levels;

	pyramid->images_pyramid = (BITMAP**)malloc(sizeof(BITMAP*)* pyramid->max_pyramid_level);
  pyramid->images_pyramid[0] = image;
	pyramid->masks_pyramid = (BITMAP**)malloc(sizeof(BITMAP*)* pyramid->max_pyramid_level);
  pyramid->masks_pyramid[0] = mask;
  pyramid->inv_masks_pyramid = (BITMAP**)malloc(sizeof(BITMAP*)* pyramid->max_pyramid_level);
  pyramid->inv_masks_pyramid[0] = inverse_mask_bitmap(mask);

	for (int i=1 ; i < pyramid->max_pyramid_level ; i++) {
		pyramid->images_pyramid[i] = downscale_image(pyramid->images_pyramid[i-1]);
		pyramid->masks_pyramid[i] = translate_mask(threshold_image(downscale_image(pyramid->masks_pyramid[i-1]), 128),p->patch_w, p->patch_w);
    pyramid->inv_masks_pyramid[i] = inverse_mask_bitmap(pyramid->masks_pyramid[i]);
	}

  save_bitmap(pyramid->inv_masks_pyramid[pyramid->max_pyramid_level-1], "mask_small.bmp");

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


/*
	Use a mask where 1 is set for non masked pixels (so that the algorithm
	wouldn't attempt to find a nn for it.
*/

void inpaint_image(Params *p, BITMAP *image, BITMAP *mask, BITMAP *inv_mask)
{
	/*
		1. Create an initial nnf by setting each patch's neighbour as itself, if it
		does not contain a mask.
		2. Let the NN algorithm do it's job for the masked part of the image
		3. Paint the masked parts of the image using the nnf
	 */

	RegionMasks *amask = mask ? new RegionMasks(p, inv_mask, /*full=*/1): NULL;

	// RegionMasks segments the image to several regions, which do not intermix
  // when finding nn

  BITMAP *ann = init_nn(p, image, image, mask, /*amaskm=*/ NULL, amask,
                        /*trip_patch=*/ 1, NULL, NULL);

  RecomposeParams *rp = new RecomposeParams();
  copy_unmasked_nnf_regions(p, image, amask, ann);
  BITMAP *annd = init_dist(p, image, image, ann, inv_mask, NULL, amask);
  minnn(p, image, image, ann, annd, /*ann_prev=*/ ann, mask, 0, 0, rp, NULL, amask, p->cores);

  BITMAP *ans = vote(p, image, ann, NULL, mask, NULL, 1, 0, amask, NULL, image, NULL, 0);

  // save_bitmap(annd, "annd.bmp");
  save_bitmap(ans, "ans.bmp");

  delete rp;

}


void copy_unmasked_nnf_regions(Params *p, BITMAP *image, RegionMasks *amask, BITMAP *ann)
{
  if (!amask) {
    fprintf(stderr, "amask must be non NULL");
    exit(1);
  }

  Box box = get_abox(p, image, amask);
  int ystart = box.ymin, yend = box.ymax, xstart = box.xmin, xend = box.xmax;

  // Box mask_box = amask->box[1];

  for (int y = ystart ; y < yend ; y++) {
    int *annr = (int *) ann->line[y];
    for (int x = xstart ; x < xend ; x++) {
      if (!is_point_in_box(y, x, box, p->patch_w))
        annr[x] = XY_TO_INT(x, y);
    }
  }
}

int is_point_in_box(int y, int x, Box box, int border)
{
  return x > box.xmin-border+1 && x < box.xmax+border-1 && y > box.ymin-border+1
    && y < box.ymax+border-1;
}

BITMAP *inverse_mask_bitmap(BITMAP *mask)
{
  BITMAP *inv_mask = create_bitmap(mask->w, mask->h);
  for (int y = 0 ; y < mask->h ; y++) {
    int *mask_row = (int *) mask->line[y];
    int *inv_mask_row = (int *) inv_mask->line[y];
    for (int x = 0 ; x < mask->w ; x++) {
        inv_mask_row[x] = (mask_row[x]&0x00ffffff)^0x00ffffff;
    }
  }

  return inv_mask;

}


void save_dist_bitmap(BITMAP *dist, const char * file_path)
{
  BITMAP *saved_bitmap = create_bitmap(dist->w, dist->h);
  for (int y = 0 ; y < dist->h ; y++) {
    int *dist_row = (int *) dist->line[y];
    int *saved_row = (int *) saved_bitmap->line[y];
    for (int x = 0 ; x < dist->w ; x++) {
      unsigned char pixel_val = max(255, dist_row[x]);
      saved_row[x] = pixel_val<<16|pixel_val<<8|pixel_val;
    }
  }

  save_bitmap(saved_bitmap, file_path);
  destroy_bitmap(saved_bitmap);
}


BITMAP *threshold_image(BITMAP *image, unsigned char threshold)
{
  int ymax = image->h, xmax = image->w;
  for (int y = 0 ; y < ymax ; y++) {
    int *row = (int *)image->line[y];
    for (int x = 0 ; x < xmax ; x++) {
      if ((row[x]&255) < threshold)
        row[x] = 0;
      else
        row[x] = 0xffffffff;
    }
  }

  return image;

}


BITMAP *translate_mask(BITMAP *mask, unsigned int translation, unsigned int min_edge_distance)
{
  int ymax = mask->h, xmax = mask->w;
  for (int y = 0 ; y < ymax ; y++) {
    int *row = (int *)mask->line[y];
    for (int x = 0 ; x < xmax ; x++) {
      int nx = x, ny = y;

      // TODO: decide if we should add more conditions, such as checking if the
      //       masked pixel is close to the lower or right edge of the image
      // if (x+translation < xmax && x+translation > xmax-1-min_edge_distance && row[x+translation]) {
      if (row[x] && xmax-1-x < min_edge_distance) {
        // row[x] = row[x+translation];
        nx = xmax - min_edge_distance;
      }

      // if (y+translation < ymax && y+translation > ymax-1-min_edge_distance && ((int *)mask->line[y+translation])[x]) {
      //   row[x] = ((int *)mask->line[y+translation])[x];
      // }

      if (((int *)mask->line[y])[x] && ymax-1-y < min_edge_distance ) {
        ny = ymax - min_edge_distance;
      }

      ((int *)mask->line[ny])[nx] = ((int *)mask->line[y])[x];
    }
  }

  return mask;
}
