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
struct UpscaleInpintedImageRetVal;
struct MaskTransformationParameters;

void free_pyramid(Pyramid *pyramid);
void build_pyramid(Params *p, Pyramid * pyramid, BITMAP *image, BITMAP *mask);
/*
  If the function receives an empty ann, it allocates a new BITMAP.
  The calculates NN fields, and inpaints an image. The returned NN field is
  returned via the returned struct, as well as the inpainted image.
*/
BITMAP *inpaint_image(Params *p, BITMAP *image, BITMAP *mask, RegionMasks *amask, BITMAP *ann=NULL);
void copy_unmasked_nnf_regions(Params *p, BITMAP *image, RegionMasks *amask, BITMAP *ann);
int is_point_in_box(int y, int x, Box box, int border);
BITMAP *inverse_mask_bitmap(BITMAP *mask);
BITMAP *threshold_image(BITMAP *image, unsigned char threshold);
/*
  This function upscales the input image, to the resolution indicated by the
  to_level image in the image pyramid. The image parts which are more than
  p->inpaint_border from a masked pixel are copied from their counterpart in the
  pyramid, thus obtaining a higher resolution copy where possible.
  The ann is also upscaled. Nearest neighbours pixels, which are more than
  p->inpaint_border from their closest masked pixel, are set to their location,
  indicating the NN patchs are themselves.

  image and ann arguments are freed and reallocated.
*/
// TODO: return image and ann using Upscaleinpintedimageretval
UpscaleInpintedImageRetVal upscale_image_nn(Params *p, Pyramid *pyramid, BITMAP *image,
                      BITMAP *ann, int to_level);

/*
  This function goes over the mask, looking for masked pixels which are at most
  min_edge_distance from the right or lower border, and moves it upwards or
  left (or both).

  Returns a new mask.

*/
BITMAP *transform_mask(Params *p, BITMAP *mask, unsigned int min_edge_distance);
void draw_box_around_mask_point(BITMAP *mask, int y, int x, int border);

typedef struct MaskTransformationParameters
{
  int min_edge_distance;
  int mask_border_width;
} MaskTransformation;

typedef struct UpscaleInpintedImageRetVal
{
  BITMAP *image;
  BITMAP *ann;
} InpaintImageRetVal;


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
  char filename[128];
  Pyramid pyramid;
	build_pyramid(p, &pyramid, a, mask);


  BITMAP *ann = NULL, *inpainted_image = NULL;
  int min_level = p->max_inpaint_levels ? pyramid.max_pyramid_level-1 - p->max_inpaint_levels : 0;

  for (int level = pyramid.max_pyramid_level-1 ; level>=min_level ; level--) {
    printf("inpaint: In level %d\n", level);

    BITMAP *timage = pyramid.images_pyramid[level];
    BITMAP *tmask = pyramid.masks_pyramid[level];
    BITMAP *tinv_mask = pyramid.inv_masks_pyramid[level];
    RegionMasks *tamask = new RegionMasks(p, tinv_mask, /*full=*/0);

    if (level == pyramid.max_pyramid_level-1) {
      ann = init_nn(p, timage, timage, /*bmask=*/tmask, /*region_masks=*/NULL,
                    /*amask=*/ tamask, /*trim_patch=*/ 1, NULL, NULL);
      copy_unmasked_nnf_regions(p, timage, tamask, ann);
      inpainted_image = inpaint_image(p, timage, tmask, tamask, ann);
      snprintf(filename, sizeof(filename)/sizeof(char),
               "inpainted_image_level_%d.png", level);
      save_bitmap(inpainted_image, filename);

    }
    else {
      UpscaleInpintedImageRetVal upscaled = upscale_image_nn(p, &pyramid, inpainted_image, ann, level);
      snprintf(filename, sizeof(filename)/sizeof(char),
               "upscaled_image_level_%d.png", level);
      save_bitmap(upscaled.image, filename);
      destroy_bitmap(inpainted_image);
      destroy_bitmap(ann);

      ann = upscaled.ann;
      // TODO: copy the unmasked pixels from the original image. vote() does not do that.
      copy_unmasked_nnf_regions(p, timage, tamask, ann);
      inpainted_image = inpaint_image(p, upscaled.image, tmask, tamask, ann);
      destroy_bitmap(upscaled.image);
      snprintf(filename, sizeof(filename)/sizeof(char),
               "inpainted_image_level_%d.png", level);
      save_bitmap(inpainted_image, filename);
    }
    delete tamask;
  }


  destroy_bitmap(inpainted_image);
  destroy_bitmap(ann);
  free_pyramid(&pyramid);

	return NULL;
  // Inpaint the last level of the pyramid
  // Defining temporaries



  // copy_unmasked_nnf_regions(p, timage, tamask, ann);
  // BITMAP *ret_inpainted = inpaint_image(p, timage, tmask, tamask, ann);
  // delete tamask;
  // save_bitmap(ret_inpainted, "ans.bmp");

  // UpscaleInpintedImageRetVal upscaled = upscale_image_nn(p, &pyramid, ret_inpainted, ann, pyramid.max_pyramid_level-2);
  // destroy_bitmap(ret_inpainted);
  // destroy_bitmap(ann);

  // save_bitmap(ret_inpainted, "ans2.bmp");
  // timage = pyramid.masks_pyramid[pyramid.max_pyramid_level-2];
  // tmask = pyramid.masks_pyramid[pyramid.max_pyramid_level-2];
  // tinv_mask = pyramid.inv_masks_pyramid[pyramid.max_pyramid_level-2];
  // tamask = new RegionMasks(p, tinv_mask, /*full=*/0);
  // BITMAP *ret_inpainted2 = inpaint_image(p, ret_inpainted, tmask, tamask, ann);
  // save_bitmap(ret_inpainted2, "ans3.bmp");
  // destroy_bitmap(ret_inpainted);


  // destroy_region_masks(tamask);
}


void build_pyramid(Params *p, Pyramid * pyramid, BITMAP *image, BITMAP *mask)
{

	// BITMAP *image = pyramid->image_initial;
	int max_possible_levels = (int)log2((double)std::min(image->h, image->w)/p->patch_w) + 1;
  BITMAP *scaled_mask = NULL;

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

  scaled_mask = pyramid->masks_pyramid[0];

	for (int i=1 ; i < pyramid->max_pyramid_level ; i++) {
		pyramid->images_pyramid[i] = downscale_image(pyramid->images_pyramid[i-1]);
    char filename[128];
    snprintf(filename, 128, "downscaled_image_%d.png", i);
    save_bitmap(pyramid->images_pyramid[i], filename);
    // save_bitmap(downscale_image(pyramid->masks_pyramid[i-1]), filename);
		// pyramid->masks_pyramid[i] = transform_mask(p, threshold_image(downscale_image(pyramid->masks_pyramid[i-1]), 128), p->patch_w);
    BITMAP *prev_mask = scaled_mask;
    scaled_mask = threshold_image(downscale_image(prev_mask), 128);
    if (i >= 2) {
      destroy_bitmap(prev_mask); prev_mask = NULL;
    }
    pyramid->masks_pyramid[i] = transform_mask(p, scaled_mask, p->patch_w);
    pyramid->inv_masks_pyramid[i] = inverse_mask_bitmap(pyramid->masks_pyramid[i]);
    // {
    //   BITMAP *tmp = transform_mask(p, pyramid->masks_pyramid[i], p->patch_w);

    //   save_bitmap(tmp, filename);
    //   destroy_bitmap(tmp);
    // }
    snprintf(filename, 128, "transformed_mask_level_%d.png", i);
    save_bitmap(pyramid->masks_pyramid[i], filename);
	}

  destroy_bitmap(scaled_mask);

  save_bitmap(pyramid->inv_masks_pyramid[pyramid->max_pyramid_level-1], "mask_small.bmp");

}


UpscaleInpintedImageRetVal upscale_image_nn(Params *p, Pyramid *pyramid, BITMAP *image,
                                 BITMAP *ann, int to_level)
{
  UpscaleInpintedImageRetVal ans = {};
  int h = pyramid->images_pyramid[to_level]->h, w = pyramid->images_pyramid[to_level]->w;

  // Some sanity checks
  if ((image)->h >= h || (image)->w >= w) {
    fprintf(stderr, "upscale_image: the target image must be of higher "\
            "resolution");
    exit(1);
  }

  // 1. upscale image
  BITMAP *nimage = scale_image(image, h, w);
  // Original image is unnecessary now
  // destroy_bitmap(image); image = NULL;

  BITMAP *nann = create_bitmap(w, h);

  RegionMasks rg(p, pyramid->inv_masks_pyramid[to_level], 0, NULL);
  // Assuming masks hold the value 0xff
  Box b = rg.box[255];

  // 2. copy high resolution parts from the the to_level image, set the new mask from
  for (int y = 0; y < h; y++) {
    int *nrow = (int *) nimage->line[y];
    int *orig_row = (int *)pyramid->images_pyramid[to_level]->line[y];
    int *annrow = (int *)(ann)->line[y/2];
    int *nannrow = (int *)nann->line[y];
    for (int x = 0; x < w; x++) {
      if (!is_point_in_box(y, x, b, p->inpaint_border)) {
        nrow[x] = orig_row[x];
        nannrow[x] = XY_TO_INT(x, y);
      }
      else {
        nannrow[x] = XY_TO_INT(INT_TO_X(annrow[x/2])*2, INT_TO_Y(annrow[x/2])*2);
      }
    }
  }

  image = nimage;
  // destroy_bitmap(ann);
  ans.ann = nann;
  ans.image = nimage;
  return ans;
}



void free_pyramid(Pyramid *pyramid)
{
	// TODO: complete
}


/*
	Use a mask where 1 is set for non masked pixels (so that the algorithm
	wouldn't attempt to find a nn for it.
*/

BITMAP *inpaint_image(Params *p, BITMAP *image, BITMAP *mask, RegionMasks *amask, BITMAP *ann)
{
	/*
		1. Create an initial nnf by setting each patch's neighbour as itself, if it
		does not contain a mask.
		2. Let the NN algorithm do it's job for the masked part of the image
		3. Paint the masked parts of the image using the nnf
	 */
	// RegionMasks segments the image to several regions, which do not intermix
  // when finding nn

  if (!ann) {
    ann = init_nn(p, image, image, /*bmask=*/mask, /*region_masks=*/NULL,
                  /*amask=*/ amask, /*trim_patch=*/ 1, NULL, NULL);
  }

  RecomposeParams *rp = new RecomposeParams();

  // TODO: consider using inv_mask for bmask, because currently the function
  //       would fill in every non-identical mask pixel between a and b with INT_MAX
  BITMAP *annd = init_dist(p, image, image, ann, /*bmask=*/mask, NULL, amask);
  nn(p, image, image, ann, annd, amask, /*bmask=*/mask, 0, 0, rp, 0, 0, 0,
     /*region_masks=*/NULL, p->cores, NULL, NULL);
  minnn(p, image, image, ann, annd, /*ann_prev=*/ ann, mask, /*level=*/0, 0, rp,
        NULL, amask, p->cores);
  delete rp;

  BITMAP *inpainted_image = vote(p, image, ann, NULL, mask, NULL, 1, 0, amask, NULL, image, NULL, 0);
  // save_bitmap(annd, "annd.bmp");
  return inpainted_image;
}


void copy_unmasked_nnf_regions(Params *p, BITMAP *image, RegionMasks *amask, BITMAP *ann)
{
  if (!amask) {
    fprintf(stderr, "amask must be non NULL");
    exit(1);
  }

  Box boundries_box = amask->box[255];
  Box mask_box = amask->box[0]; // 0 denotes the unmasked regions, but because
                                // we want the algorithm in nn function to go
                                // over our unmasked region, we inverted the
                                // mask image
  int ystart = 0, yend = image->h, xstart = 0, xend = image->w;

  // Box mask_box = amask->box[1];

  for (int y = ystart ; y < yend ; y++) {
    int *annr = (int *) ann->line[y];
    for (int x = xstart ; x < xend ; x++) {
      if (!is_point_in_box(y, x, mask_box, p->patch_w))
        annr[x] = XY_TO_INT(x, y);
    }
  }
}

int is_point_in_box(int y, int x, Box box, int border)
{
  return x >= box.xmin && x < box.xmax+border-1 && y >= box.ymin
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


BITMAP *transform_mask(Params *p, BITMAP *mask, unsigned int min_edge_distance)
{
  int ymax = mask->h, xmax = mask->w;
  BITMAP *ans = new BITMAP(*mask);
  std::vector<int> mask_points;
  for (int y = 0 ; y < ymax ; y++) {
    int *row = (int *)mask->line[y];
    int *ansrow = (int *)ans->line[y];
    for (int x = 0 ; x < xmax ; x++) {
      int nx = x, ny = y;

      if (row[x] && xmax-1-x < min_edge_distance && xmax-min_edge_distance >= 0) {
        nx = xmax - min_edge_distance;
      }
      else if (row[x] && x-min_edge_distance/2 >= 0) {
        // We wish the masked pixel to be in the middle of the box. Without this
        // line and it's correspondent in y, the box would stretch to the
        // lower-right edge of the masked pixel
        nx = x-min_edge_distance/2;
      }


      if (row[x] && ymax-1-y < min_edge_distance && ymax-min_edge_distance >= 0) {
        ny = ymax - min_edge_distance;
      }
      else if (row[x] && y-min_edge_distance/2 >= 0) {
        ny = y-min_edge_distance/2;
      }

      if (row[x]) {
        mask_points.push_back(XY_TO_INT(nx, ny));
      }

      ((int *)ans->line[ny])[nx] = ((int *)mask->line[y])[x];
    }
  }

  for (std::vector<int>::iterator iter = mask_points.begin() ;
       iter != mask_points.end() ; ++iter) {
    draw_box_around_mask_point(ans, INT_TO_X(*iter), INT_TO_Y(*iter), p->inpaint_border);
  }


  // RegionMasks rm (p, mask);



  // for (int y = 0 ; y < ymax ; y++) {
  //   int *row = (int *)ans->line[y];
  //   int *ansrow = (int *)ans->line[y];
  //   for (int x = 0 ; x < xmax ; x++) {
  //     // if (is_point_in_box(y, x, rm.box[255], p->inpaint_border)) {
  //     //   row[x] = 0xffffff;
  //     // }
  //     if (row[x]) {
  //       draw_box_around_mask_point(ans, y, x, p->inpaint_border);
  //     }
  //   }
  // }


  // Ugly, I know
  // TODO: Make something nicer, use move semantics or something
  // delete [] mask->line;
  // delete [] mask->data;
  // mask->line = ans->line;
  // mask->data = ans->data;
  // delete ans;

  return ans;
}


void draw_box_around_mask_point(BITMAP *mask, int y, int x, int border)
{
  for (int ty=y-border ; ty <= y+border ; ty++) {
    for (int tx=x-border ; tx <= x+border ; tx++) {
      if (tx >= 0 && tx < mask->w && ty >= 0 && ty < mask->h)
        ((int*)mask->line[ty])[tx] = 0xffffff;
    }
  }

  //   if (ty >= 0 && ty < mask->h)
  //     ((int*)mask->line[ty])[x] = 0xffffff;
  // }

}
