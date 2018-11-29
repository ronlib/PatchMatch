#include <cmath>
#include <algorithm>
#include <unistd.h>
#include <cassert>
#include <climits>

#include "nn.h"
#include "lua_inpaint.h"
#include "allegro_emu.h"
#include "inpaint.h"

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
                      RegionMasks *rm_inv_inpainted_patch_mask, int level,
                      BITMAP *ann=NULL, bool add_completion=false);
/*
  This function copies each unmasked pixel from orig_image to image.
*/
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
BITMAP *create_transformed_mask(Params *p, BITMAP *mask, int min_edge_distance,
                       int mask_center_offset, int border_size, int transform_operation);
void draw_box_around_mask_point(BITMAP *mask, int x, int y, int border);
void visualize_nnf(BITMAP *nn, const char* filename);
BITMAP* create_black_box(int h, int w);
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
  BITMAP **masks_pyramid; // Not used directly. Made using
                          // downscaling of the original mask
  // When pixel value is 1, do not use the patch at that location for coherence
  BITMAP **bmasks;
  // When pixel value is 1, we wish to find a match for this patch when inpainting
  BITMAP **inpainted_patch_masks;
  // The inverse of inpainted_patch_masks, used in init_nn and other functions.
  // When the value is 0, we wish to inpaint that patch
  BITMAP **inv_inpainted_patch_masks;

  // All black image for marking regions we wish to inpaint in case of extended
  // inpainting
  BITMAP **black;

  int max_pyramid_level;

  Pyramid(int max_pyramid_level_=0): max_pyramid_level(max_pyramid_level_) {}
};


BITMAP *inpaint(Params *p, BITMAP *a, BITMAP *mask, bool add_completion)
{
  char filename[128];
  Pyramid pyramid;
  build_pyramid(p, &pyramid, a, mask);


  BITMAP *ann = NULL, *inpainted_image = NULL;
  int min_level = p->max_inpaint_levels ? pyramid.max_pyramid_level-1 - p->max_inpaint_levels : 0;

  for (int level = pyramid.max_pyramid_level-1 ; level>=min_level ; level--) {
    printf("inpaint: In level %d\n", level);

    BITMAP *image = pyramid.images_pyramid[level];
    BITMAP *inv_inpainted_patch_mask = pyramid.inv_inpainted_patch_masks[level];
    BITMAP *bmask = pyramid.bmasks[level];
    RegionMasks *
      rm_inv_inpainted_patch_mask = new RegionMasks(p, inv_inpainted_patch_mask, /*full=*/0);

    if (level == pyramid.max_pyramid_level-1) {
      ann = init_nn(p, image, image, /*bmask=*/bmask, /*region_masks=*/NULL,
                    /*amask=*/ rm_inv_inpainted_patch_mask, /*trim_patch=*/ 1, NULL, NULL);
      inpainted_image = inpaint_image(p, &pyramid,
                                      image,                         // image to inpaint
                                      rm_inv_inpainted_patch_mask,   // mask of patches to inpaint
                                      level,
                                      ann,                          // precomputed ann
                                      add_completion);
      snprintf(filename, sizeof(filename)/sizeof(char),
               "inpainted_image_level_%d.png", level);
      save_bitmap(inpainted_image, filename);

    }
    else {
      UpscaleInpintedImageRetVal upscaled =
        upscale_image_nn(p, &pyramid, inpainted_image, ann, level);
      snprintf(filename, sizeof(filename)/sizeof(char),
               "upscaled_image_level_%d.png", level);
      save_bitmap(upscaled.image, filename);
      destroy_bitmap(ann); ann = 0;
      ann = upscaled.ann;

      // Similar to what younesse did see "Space-Time Video Completion" - page 5),
      // but different in that we copy the unmasked pixels for every pyramid level
      if (/*level == 0*/1) {
        destroy_bitmap(inpainted_image); inpainted_image = 0;
        // copy_unmasked_nnf_regions(p, timage, tamask, ann);
        inpainted_image = inpaint_image(p, &pyramid, upscaled.image, rm_inv_inpainted_patch_mask, level, ann, add_completion);
      }
      // else {
      //   BITMAP *tmp = inpainted_image;
      //   inpainted_image = scale_image(inpainted_image,
      //                                 pyramid.images_pyramid[level]->h,
      //                                 pyramid.images_pyramid[level]->w);
      //   inpainted_image = inpaint_image(p, &pyramid, inpainted_image, rm_inv_inpainted_patch_mask, level, ann);
      //   destroy_bitmap(tmp);
      // }
      destroy_bitmap(upscaled.image);
      snprintf(filename, sizeof(filename)/sizeof(char),
               "inpainted_image_level_%d.png", level);
      save_bitmap(inpainted_image, filename);
    }
    delete rm_inv_inpainted_patch_mask;
  }


  destroy_bitmap(inpainted_image);
  destroy_bitmap(ann);
  free_pyramid(&pyramid);

  return NULL;
}


void build_pyramid(Params *p, Pyramid * pyramid, BITMAP *image, BITMAP *mask)
{

  int max_possible_levels = (int)log2((double)std::min(image->h, image->w)/p->patch_w) + 1;
  BITMAP *scaled_mask = NULL;
  char filename[128];
  int level = 0;

  if (pyramid->max_pyramid_level != 0 && max_possible_levels > pyramid->max_pyramid_level)
    max_possible_levels = pyramid->max_pyramid_level;
  else
    pyramid->max_pyramid_level = max_possible_levels;

  pyramid->images_pyramid =
    (BITMAP**)malloc(sizeof(BITMAP*)* pyramid->max_pyramid_level);
  pyramid->masks_pyramid = (BITMAP**)malloc(sizeof(BITMAP*)* pyramid->max_pyramid_level);
  pyramid->bmasks = (BITMAP**)malloc(sizeof(BITMAP*)* pyramid->max_pyramid_level);
  pyramid->inpainted_patch_masks =
    (BITMAP**)malloc(sizeof(BITMAP*)* pyramid->max_pyramid_level);
  // pyramid->inv_masks_pyramid = (BITMAP**)malloc(sizeof(BITMAP*)* pyramid->max_pyramid_level);
  pyramid->inv_inpainted_patch_masks =
    (BITMAP**)malloc(sizeof(BITMAP*)* pyramid->max_pyramid_level);
  pyramid->black = (BITMAP**)malloc(sizeof(BITMAP*)* pyramid->max_pyramid_level);

  for (level=0 ; level < pyramid->max_pyramid_level ; level++) {
    BITMAP *scaled_mask_copy = 0;
    if (level == 0) {
        pyramid->images_pyramid[level] = image;
        pyramid->masks_pyramid[level] = new BITMAP(*mask);
          // create_transformed_mask(p, mask, p->patch_w, p->patch_w/2, p->inpaint_border, ADD_BORDERS);
        // In order for this confition to be compatible with the rest of the loop
        scaled_mask_copy = scaled_mask = pyramid->masks_pyramid[level];
    }
    else {
      pyramid->images_pyramid[level] = downscale_image(pyramid->images_pyramid[level-1]);
      // Scaling down the original mask, and not the thresholded one
      /* BITMAP *prev_mask = scaled_mask; */

      // TODO: restore previous line if the following change doesn't work out
      scaled_mask = downscale_image(/*scaled_mask*/ pyramid->masks_pyramid[level-1]);
      /* destroy_bitmap(prev_mask); prev_mask = 0; */

      /* scaled_mask_copy = new BITMAP(*scaled_mask); */
      threshold_image(scaled_mask, p->mask_threshold);
      pyramid->masks_pyramid[level] = scaled_mask;
        // create_transformed_mask(p, scaled_mask_copy, p->patch_w,
        //                p->patch_w/2, p->inpaint_border, ADD_BORDERS);
    }

    snprintf(filename, 128, "downscaled_image_%d.png", level);
    save_bitmap(pyramid->images_pyramid[level], filename);

    snprintf(filename, 128, "mask_level_%d.png", level);
    save_bitmap(pyramid->masks_pyramid[level], filename);

    pyramid->bmasks[level] =
      create_transformed_mask(p,
                              scaled_mask, // mask
                              0, //min_edge_distance
                              p->patch_w/2, // mask_center_offset
                              p->patch_w/2, // border_size
                              ADD_BORDERS | CENTER_MASK); // transform_operation
    /* if (level != 0) {
      destroy_bitmap(scaled_mask_copy);
    } */

    snprintf(filename, 128, "bmask_%d.png", level);
    save_bitmap(pyramid->bmasks[level], filename);

    // pyramid->inv_masks_pyramid[level] = inverse_mask_bitmap(pyramid->masks_pyramid[level]);
    pyramid->inpainted_patch_masks[level] =
      create_transformed_mask(p,
                              pyramid->masks_pyramid[level], // mask
                              p->patch_w,                    // min_edge_distance
                              p->patch_w/2,                  // mask_center_offset
                              p->inpaint_border,             // border_size
                              ADD_BORDERS | TRANSLATE_FROM_EDGES | CENTER_MASK);
    pyramid->inv_inpainted_patch_masks[level] =
      inverse_mask_bitmap(pyramid->inpainted_patch_masks[level]);

    snprintf(filename, 128, "inpainted_patch_masks_level_%d.png", level);
    save_bitmap(pyramid->inpainted_patch_masks[level], filename);

    pyramid->black[level] = create_black_box(pyramid->bmasks[level]->h, pyramid->bmasks[level]->w);
    snprintf(filename, 128, "black_level_%d.png", level);
    save_bitmap(pyramid->black[level], filename);
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
  // 2. copy high resolution parts from the the to_level image, set the new mask from
  copy_unmasked_regions(p, nimage, // new image
                        pyramid->masks_pyramid[to_level], // mask
                        pyramid->images_pyramid[to_level]); // original image

  BITMAP *nann = create_bitmap(w, h);

  // TODO: Consider removing this and then the whole function (just use
  //       copy_unmasked_regions). Not sure if there is a meaning to upscaling
  //       the NNF. PatchMatch will do pretty good without it.
  int h_ = std::min(h, image->h*2);
  int w_ = std::min(h, image->w*2);
  for (int y = 0; y < h_; y++) {
    int *annrow = (int *)(ann->line[y/2]);
    int *nannrow = (int *)nann->line[y];
    int *inv_inpainted_patch_mask = (int *) pyramid->inv_inpainted_patch_masks[to_level]->line[y];
    for (int x = 0; x < w_; x++) {
      // Wherever there is no mask, NN for each neighbour is itself. Otherwise,
      // we set it to the scaled previous NN
      if (inv_inpainted_patch_mask[x]) {
        nannrow[x] = XY_TO_INT(x, y);
      }
      else {
        nannrow[x] = XY_TO_INT(INT_TO_X(annrow[x/2])*2, INT_TO_Y(annrow[x/2])*2);
      }
    }
  }

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
                      RegionMasks *rm_inv_inpainted_patch_mask, int level,
                      BITMAP *ann, bool add_completion)
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
  BITMAP *bmask = pyramid->bmasks[level];
  BITMAP *inv_bmask = inverse_mask_bitmap(bmask);
  BITMAP *black = pyramid->black[level];
  RegionMasks *rm_black = new RegionMasks(p, black);
  RegionMasks *rm_bmask = new RegionMasks(p, bmask);
  RegionMasks *rm_inv_bmask = new RegionMasks(p, inv_bmask);
  BITMAP *inpainted_image = image;
  BITMAP *bnnd = NULL, *bnn = NULL;
  rm_inv_inpainted_patch_mask = 0;
  // TODO: remove
  char filename[128] = {0};

  if (!ann) {
    ann = init_nn(p, image, orig_image, /*bmask=*/bmask, /*region_masks=*/NULL,
                  /*amask=*/ rm_inv_inpainted_patch_mask, /*trim_patch=*/ 1,
                  /*ann_window=*/NULL, /*awinsize=*/NULL);
  }

  if (add_completion) {
    bnn = init_nn(p, orig_image, image, /*bmask=*/black, /*region_masks=*/NULL,
                  /*amask=*/ rm_bmask, /*trim_patch=*/ 1,
                  /*ann_window=*/NULL, /*awinsize=*/NULL);

  }

  RecomposeParams *rp = new RecomposeParams();

  BITMAP *annd = init_dist(p,
                           image,
                           orig_image,            // b image
                           ann,                   // ann
                           bmask,                 // bmask
                           NULL,                  // region_masks, seperating regions
                           // We use rm_black in case of add_completion, because
                           // only when completion term is added, we can be sure
                           // that no artifacts will be added to the inpainted
                           // image
                           /*amask=*/add_completion ? rm_black : rm_inv_inpainted_patch_mask);
  if (add_completion) {
    bnnd = init_dist(p,
                     orig_image,
                     image,                 // b image
                     bnn,                   // ann
                     black,                 // bmask
                     NULL,                  // region_masks, seperating regions
                     /*amask=*/rm_bmask);
  }
  for(int i = 0 ; i < (pyramid->max_pyramid_level+10 - level) ; i++) {
    nn(p, inpainted_image, orig_image, ann, annd,
       /*amask=*/add_completion ? rm_black : rm_inv_inpainted_patch_mask,
       /*bmask=*/bmask, 0, 0, rp, 0, 0, 0, /*region_masks=*/NULL, p->cores, NULL,
       NULL);
    minnn(p, inpainted_image, orig_image, ann, annd, /*ann_prev=*/ ann, /*bmask=*/bmask, /*level=*/0, 0, rp,
          /*region_masks=*/NULL, /*amask=*/add_completion ? rm_black : rm_inv_inpainted_patch_mask, p->cores);
    snprintf(filename, 128, "ann_level_%d_iter_%d.png", level, i);
    visualize_nnf(annd, filename);

    if (add_completion) {
      nn(p, /*a=*/orig_image, /*b=*/inpainted_image, /*ann=*/bnn, /*annd=*/bnnd,
         /*amask=*/rm_bmask, /*bmask=*/black, /*level=*/0, /*em_iter=*/0, rp, 0,
         0, /*region_masks=*/0,
         /*region_masks=*/NULL, p->cores, NULL, NULL);
      minnn(p, orig_image, inpainted_image, bnn, bnnd, /*ann_prev=*/ bnn, /*bmask=*/bmask, /*level=*/0, 0, rp,
            /*region_masks=*/NULL, /*amask*/rm_bmask, p->cores);
      snprintf(filename, 128, "bnn_level_%d_iter_%d.png", level, i);
      // visualize_nnf(bnnd, filename);

    }

    BITMAP *temp = inpainted_image;
    inpainted_image = vote(p, orig_image, ann, /*bnn=*/bnn, /*bmask=*/bmask, /*bweight=*/NULL,
                           /*coherence_weight=*/1, /*complete_weight=*/add_completion ? 1 : 0,
                           /*amask=*/add_completion ? rm_black : rm_inv_inpainted_patch_mask, /*aweight=*/NULL,
                           /*ainit=*/image, /*region_masks=*/NULL, /*aconstraint=*/0,
                           /*mask_self_only=*/0);
    snprintf(filename, 128, "inpainted_image_level_%d_iter_%d.png", level, i);
    save_bitmap(inpainted_image, filename);
    if (i > 0)
      destroy_bitmap(temp);

  }
  // It is unnecessary to copy the unmasked regions, as they are copied during
  // the voting stage, because ann has correct values for unmasked regions
  // copy_unmasked_regions(p, inpainted_image, pyramid->masks_pyramid[level], orig_image);
  printf("Finished inpaint_image, add_completion=%d\n", add_completion);

  delete rm_black;
  delete rm_bmask;
  delete rp;
  return inpainted_image;
}


void copy_unmasked_regions(Params *p, BITMAP *image, BITMAP *mask, BITMAP *orig_image)
{
  if (!mask) {
    fprintf(stderr, "amask must be non NULL");
    exit(1);
  }

  int ystart = 0, yend = image->h, xstart = 0, xend = image->w;

  for (int y = ystart ; y < yend ; y++) {
    int *row = (int *) image->line[y];
    int *origr = (int *) orig_image->line[y];
    int *maskr = (int *) mask->line[y];
    for (int x = xstart ; x < xend ; x++) {
      if (!maskr[x]) {
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


BITMAP *create_transformed_mask(Params *p, BITMAP *mask, int min_edge_distance,
                       int mask_center_offset, int border_size, int transform_operation)
{
  int ymax = mask->h, xmax = mask->w;
  BITMAP *ans = create_bitmap(mask->w, mask->h);
  clear(ans);
  std::vector<int> mask_points;
  for (int y = 0 ; y < ymax ; y++) {
    unsigned int *row = (unsigned int *)mask->line[y];
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
  }

  int tborder_size = (transform_operation & ADD_BORDERS) ? border_size : 0;

  for (std::vector<int>::iterator iter = mask_points.begin() ;
       iter != mask_points.end() ; ++iter) {
    draw_box_around_mask_point(ans, INT_TO_X(*iter), INT_TO_Y(*iter), tborder_size);
  }

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
}

void visualize_nnf(BITMAP *nn, const char* filename) {
  BITMAP *visualized_nnf = create_bitmap(nn->w, nn->h);
  double max_distance = sqrt(nn->w*nn->w+nn->h*nn->h);
  for (int y = 0 ; y < nn->h ; y++) {
    for (int x = 0 ; x < nn->w ; x++) {
      int *nnf = (int*)&(((nn->line[y])[4*x]));
      int dy = INT_TO_Y(*nnf), dx = INT_TO_X(*nnf);
      unsigned char distance = (unsigned char)255*(sqrt((double)((dy-y)*(dy-y) + (dx-x)*(dx-x)))/max_distance);
      *(int*)&((visualized_nnf->line[y])[4*x]) = distance | distance << 8 | distance << 16 | distance << 24;
    }
  }

  save_bitmap(visualized_nnf, filename);
  destroy_bitmap(visualized_nnf);
}

BITMAP* create_black_box(int h, int w) {
  BITMAP *black_img = create_bitmap(w, h);

  for (int y = 0 ; y < h ; y++) {
    for (int x = 0 ; x < w ; x++) {
      ((int*) black_img->line[y])[x] = 0;
    }
  }

  return black_img;
}
