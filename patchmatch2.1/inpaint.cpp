#include <cmath>
#include <algorithm>
#include <unistd.h>
#include <cassert>

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
BITMAP *inpaint_image(Params *p, Pyramid *pyramid, BITMAP *image,
                      RegionMasks *amask, int level, BITMAP *ann=NULL);
void copy_unmasked_regions(Params *p, BITMAP *image, BITMAP *mask, BITMAP *orig_image);
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

enum TransformOperation
  {
   ADD_BORDERS = 1 << 0,
   TRANSLATE_FROM_EDGES = 1 << 1,
   CENTER_MASK = 1 << 2
  };

/*
  This function goes over the mask, looking for masked pixels which are at most
  min_edge_distance from the right or lower border, and moves it upwards or
  left (or both).
  Next, it goes over each masked pixel, moves it half a patch up and left, so
  it sould be at the center of the patch for which a nearest neighbour will be
  searched for. This translation must be accounted for in upscale_image_nn.

  @param translate - should each masked pixel be translated by p->patch_w/2

  Returns a new mask.
*/
BITMAP *transform_mask(Params *p, BITMAP *mask, int min_edge_distance,
                       int mask_center_offset, int border_size, int transform_operation);
void draw_box_around_mask_point(BITMAP *mask, int x, int y, int border);

// #define CENTER_MASK_PIXEL(p, position) position-p->patch_w/2
#define CENTER_MASK_PIXEL(offset, position) position-offset
// #define UNCENTER_MASK_PIXEL(p, position) position+p->patch_w/2
#define UNCENTER_MASK_PIXEL(offset, position) position+offset


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
  BITMAP **bmasks_pyramid;
  BITMAP **transformed_masks_pyramid;
  // TODO: decide if needed
  BITMAP **inv_masks_pyramid;
  BITMAP **transformed_inv_masks_pyramid;

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
    BITMAP *tmask = pyramid.transformed_masks_pyramid[level];
    BITMAP *tinv_mask = pyramid.transformed_inv_masks_pyramid[level];
    BITMAP *tbmask = pyramid.bmasks_pyramid[level];
    RegionMasks *tamask = new RegionMasks(p, tinv_mask, /*full=*/0);

    if (level == pyramid.max_pyramid_level-1) {
      ann = init_nn(p, timage, timage, /*bmask=*/tbmask, /*region_masks=*/NULL,
                    /*amask=*/ tamask, /*trim_patch=*/ 1, NULL, NULL);
      // copy_unmasked_nnf_regions(p, timage, tamask, ann);
      inpainted_image = inpaint_image(p, &pyramid, timage, tamask, level, ann);
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
      // copy_unmasked_nnf_regions(p, timage, tamask, ann);
      inpainted_image = inpaint_image(p, &pyramid, upscaled.image, tamask, level, ann);
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
  pyramid->masks_pyramid[0] = transform_mask(p, mask, p->patch_w, p->patch_w/2, p->inpaint_border, ADD_BORDERS);
  pyramid->bmasks_pyramid = (BITMAP**)malloc(sizeof(BITMAP*)* pyramid->max_pyramid_level);
  pyramid->bmasks_pyramid[0] = transform_mask(p, mask, 0, p->patch_w/2,
                                              p->patch_w/2, ADD_BORDERS | CENTER_MASK);
  pyramid->transformed_masks_pyramid = (BITMAP**)malloc(sizeof(BITMAP*)* pyramid->max_pyramid_level);
  pyramid->transformed_masks_pyramid[0] =
    transform_mask(p, mask, p->patch_w, p->patch_w/2, p->inpaint_border, ADD_BORDERS | TRANSLATE_FROM_EDGES | CENTER_MASK);
  pyramid->inv_masks_pyramid = (BITMAP**)malloc(sizeof(BITMAP*)* pyramid->max_pyramid_level);
  pyramid->inv_masks_pyramid[0] = inverse_mask_bitmap(mask);
  pyramid->transformed_inv_masks_pyramid = (BITMAP**)malloc(sizeof(BITMAP*)* pyramid->max_pyramid_level);
  pyramid->transformed_inv_masks_pyramid[0] = inverse_mask_bitmap(pyramid->transformed_masks_pyramid[0]);


  scaled_mask = pyramid->masks_pyramid[0];

	for (int i=1 ; i < pyramid->max_pyramid_level ; i++) {
    pyramid->images_pyramid[i] = downscale_image(pyramid->images_pyramid[i-1]);
    char filename[128];
    snprintf(filename, 128, "downscaled_image_%d.png", i);
    save_bitmap(pyramid->images_pyramid[i], filename);
    // save_bitmap(downscale_image(pyramid->masks_pyramid[i-1]), filename);
		// pyramid->masks_pyramid[i] = transform_mask(p, threshold_image(downscale_image(pyramid->masks_pyramid[i-1]), 128), p->patch_w);
    BITMAP *prev_mask = scaled_mask;
    scaled_mask = downscale_image(scaled_mask);
    destroy_bitmap(prev_mask);
    BITMAP *scaled_mask_copy = new BITMAP(*scaled_mask);
    // pyramid->masks_pyramid[i] = threshold_image(downscale_image(pyramid->masks_pyramid[i-1]), 128);
    threshold_image(scaled_mask_copy, p->mask_threshold);
    pyramid->masks_pyramid[i] = transform_mask(p, scaled_mask_copy,
                                               p->patch_w, p->patch_w/2, p->inpaint_border, ADD_BORDERS);
    pyramid->bmasks_pyramid[i] = transform_mask(p, scaled_mask_copy, 0, p->patch_w/2,
                                                p->patch_w/2, ADD_BORDERS | CENTER_MASK);
    destroy_bitmap(scaled_mask_copy);

    snprintf(filename, 128, "bmask_%d.png", i);
    save_bitmap(pyramid->bmasks_pyramid[i], filename);

    pyramid->inv_masks_pyramid[i] = inverse_mask_bitmap(pyramid->masks_pyramid[i]);
    pyramid->transformed_masks_pyramid[i] =
      transform_mask(p, scaled_mask, p->patch_w, p->patch_w/2, p->inpaint_border,
                     ADD_BORDERS | TRANSLATE_FROM_EDGES | CENTER_MASK);
    pyramid->transformed_inv_masks_pyramid[i] =
      inverse_mask_bitmap(pyramid->transformed_masks_pyramid[i]);
    // {
    //   BITMAP *tmp = transform_mask(p, pyramid->masks_pyramid[i], p->patch_w);

    //   save_bitmap(tmp, filename);
    //   destroy_bitmap(tmp);
    // }
    snprintf(filename, 128, "transformed_mask_level_%d.png", i);
    save_bitmap(pyramid->transformed_masks_pyramid[i], filename);
    snprintf(filename, 128, "inv_mask_level_%d.png", i);
    save_bitmap(pyramid->inv_masks_pyramid[i], filename);
	}
  destroy_bitmap(scaled_mask);
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
  Box b = rg.box[0];

  // 2. copy high resolution parts from the the to_level image, set the new mask from
  for (int y = 0; y < h; y++) {
    int *nrow = (int *) nimage->line[y];
    int *orig_row = (int *)(pyramid->images_pyramid[to_level]->line[y]);
    int *annrow = (int *)(ann)->line[y/2];
    int *nannrow = (int *)nann->line[y];
    int *invmaskr = (int *)pyramid->inv_masks_pyramid[to_level]->line[y];
    int *trnsinvmaskr = (int *) pyramid->transformed_inv_masks_pyramid[to_level]->line[y];
    for (int x = 0; x < w; x++) {

      // If the uncentered pixel of (x,y) is not masked, just copy the pixel
      if (invmaskr[x]) {
        nrow[x] = orig_row[x];
      }
      if (!invmaskr[x]) {
        invmaskr[x] = invmaskr[x];
      }

      // Wherever there is no mask, NN for each neighbour is itself. Otherwise,
      // we set it to the scaled previous NN
      if (trnsinvmaskr[x]) {
        // nrow[x] = orig_row[x];

        // ((int *)nimage->line[ny])[nx] =
        //   ((int *)pyramid->images_pyramid[to_level]->line[ny])[nx];

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

BITMAP *inpaint_image(Params *p, Pyramid *pyramid, BITMAP *image,
                      RegionMasks *amask, int level, BITMAP *ann)
{
	/*
		1. Create an initial nnf by setting each patch's neighbour as itself, if it
		does not contain a mask.
		2. Let the NN algorithm do it's job for the masked part of the image
		3. Paint the masked parts of the image using the nnf
	 */
	// RegionMasks segments the image to several regions, which do not intermix
  // when finding nn

  BITMAP *orig_image = pyramid->images_pyramid[level];
  BITMAP *mask = pyramid->masks_pyramid[level];
  BITMAP *bmask = pyramid->bmasks_pyramid[level];
  BITMAP *inpainted_image;


  if (!ann) {
    ann = init_nn(p, image, image, /*bmask=*/bmask, /*region_masks=*/NULL,
                  /*amask=*/ amask, /*trim_patch=*/ 1, NULL, NULL);
  }

  RecomposeParams *rp = new RecomposeParams();

  // TODO: consider using inv_mask for bmask, because currently the function
  //       would fill in every non-identical mask pixel between a and b with INT_MAX
  BITMAP *annd = init_dist(p, image, image, ann, /*bmask=*/bmask, NULL, amask);
  nn(p, image, image, ann, annd, amask, /*bmask=*/bmask, 0, 0, rp, 0, 0, 0,
     /*region_masks=*/NULL, p->cores, NULL, NULL);
  minnn(p, image, image, ann, annd, /*ann_prev=*/ ann, bmask, /*level=*/0, 0, rp,
        NULL, amask, p->cores);
  delete rp;

  inpainted_image = vote(p, image, ann, /*bnn=*/NULL, bmask, /*bweight=*/NULL,
                         1, 0, amask, NULL, image, NULL, 0);
  copy_unmasked_regions(p, inpainted_image, pyramid->inv_masks_pyramid[level], orig_image);
  // save_bitmap(annd, "annd.bmp");
  return inpainted_image;
}


void copy_unmasked_regions(Params *p, BITMAP *image, BITMAP *mask, BITMAP *orig_image)
{
  if (!mask) {
    fprintf(stderr, "amask must be non NULL");
    exit(1);
  }

  // Box boundries_box = amask->box[255];
  // Box mask_box = amask->box[0]; // 0 denotes the unmasked regions, but because
                                // we want the algorithm in nn function to go
                                // over our unmasked region, we inverted the
                                // mask image
  int ystart = 0, yend = image->h, xstart = 0, xend = image->w;

  // Box mask_box = amask->box[1];

  for (int y = ystart ; y < yend ; y++) {
    int *row = (int *) image->line[y];
    int *origr = (int *) orig_image->line[y];
    int *maskr = (int *) mask->line[y];
    for (int x = xstart ; x < xend ; x++) {
      // if (!is_point_in_box(y, x, mask_box, p->patch_w))
      //   annr[x] = XY_TO_INT(x, y);
      if (maskr[x]) {
        row[x] = origr[x];
      }
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


BITMAP *transform_mask(Params *p, BITMAP *mask, int min_edge_distance,
                       int mask_center_offset, int border_size, int transform_operation)
{
  int ymax = mask->h, xmax = mask->w;
  BITMAP *ans = create_bitmap(mask->w, mask->h);
  clear(ans);
  std::vector<int> mask_points;
  for (int y = 0 ; y < ymax ; y++) {
    unsigned int *row = (unsigned int *)mask->line[y];
    int *ansrow = (int *)ans->line[y];
    for (int x = 0 ; x < xmax ; x++) {
      int nx = x, ny = y;

      if ((transform_operation & TRANSLATE_FROM_EDGES) && row[x]) {
        if (xmax-1-x < min_edge_distance && xmax-min_edge_distance >= 0) {
          nx = xmax - min_edge_distance;
        }

        if (ymax-1-y < min_edge_distance && ymax-min_edge_distance >= 0) {
          ny = ymax - min_edge_distance;
        }

        if (nx != x || ny != y) {
          mask_points.push_back(XY_TO_INT(nx, ny));
          continue;
        }
      }

      if ((transform_operation & CENTER_MASK) && row[x]) {
        if (CENTER_MASK_PIXEL(mask_center_offset, x) >= 0) {
          nx = CENTER_MASK_PIXEL(mask_center_offset, x);
        }

        if (CENTER_MASK_PIXEL(mask_center_offset, y) >= 0) {
          ny = CENTER_MASK_PIXEL(mask_center_offset, y);
        }

        if (nx != x || ny != y) {
          mask_points.push_back(XY_TO_INT(nx, ny));
          continue;
        }
      }

      if ((transform_operation & ADD_BORDERS) && row[x]) {
        mask_points.push_back(XY_TO_INT(x, y));
      }

    }

      ////////////////////////////////////////////////////////////////////////////////////
      // if ((transform_operation & TRANSLATE_FROM_EDGES) && row[x] &&                  //
      //     xmax-1-x < min_edge_distance && xmax-min_edge_distance >= 0) {             //
      //   nx = xmax - min_edge_distance;                                               //
      // }                                                                              //
      // else if ((transform_operation & CENTER_MASK) && row[x] &&                      //
      //          CENTER_MASK_PIXEL(p, x) >= 0) {                                       //
      //   // We wish the masked pixel to be in the middle of the box. Without this     //
      //   // line and it's correspondent in y, the box would stretch to the            //
      //   // lower-right edge of the masked pixel                                      //
      //   nx = CENTER_MASK_PIXEL(p, x);                                                //
      // }                                                                              //
      //                                                                                //
      //                                                                                //
      // if ((transform_operation & TRANSLATE_FROM_EDGES) &&                            //
      //     row[x] && ymax-1-y < min_edge_distance && ymax-min_edge_distance >= 0) {   //
      //   ny = ymax - min_edge_distance;                                               //
      // }                                                                              //
      // else if ((transform_operation & CENTER_MASK) && row[x] &&                      //
      //          CENTER_MASK_PIXEL(p, y) >= 0) {                                       //
      //   ny = CENTER_MASK_PIXEL(p, y);                                                //
      // }                                                                              //
      //                                                                                //
      // // Adding (nx,ny) to a list of points, arround each a rectangle will be drawed //
      // if (row[x]) {                                                                  //
      //   mask_points.push_back(XY_TO_INT(nx, ny));                                    //
      // }                                                                              //
      //                                                                                //
      // // Transferred pixels should be deleted from their original location           //
      // if (nx != x || ny != y) {                                                      //
      //   ((int *)ans->line[ny])[nx] = ((int *)mask->line[y])[x];                      //
      //   ((int *)ans->line[y])[x] = 0;                                                //
      // }                                                                              //
      ////////////////////////////////////////////////////////////////////////////////////
  }

  int tborder_size = (transform_operation & ADD_BORDERS) ? border_size : 0;

  for (std::vector<int>::iterator iter = mask_points.begin() ;
       iter != mask_points.end() ; ++iter) {
    draw_box_around_mask_point(ans, INT_TO_X(*iter), INT_TO_Y(*iter), tborder_size);
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


void draw_box_around_mask_point(BITMAP *mask, int x, int y, int border)
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
