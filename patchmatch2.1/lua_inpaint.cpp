#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <libgen.h>

#include "TH/TH.h"
#include <TH/THStorage.h>
#include <TH/THTensor.h>
extern "C" {
#define LUA_LIB
#include "lua.h"
#include "lauxlib.h"
#include "luaT.h"
}
#include "vecnn.h"
#include "allegro_emu.h"
#include "knn.h"
#include "inpaint.h"
#include "lua_inpaint.h"
#include "export_def.h"
#include "patch.h"

#include "patch2vec.h"


#define MODE_IMAGE  0
#define MODE_VECB   1
#define MODE_VECF   2
#define FILE_PATH_LENGTH 512



typedef struct TorchImageStorage
{
  THByteStorage *storage;
  unsigned char *buffer;
} TorchImageStorage;


BITMAP *load_bitmap(const char *filename);
static int nn(lua_State *L);
static int vote(lua_State *L);
static int release_bitmap(lua_State *L);
int push_ann_to_stack(lua_State *L, BITMAP *ann);
void error (lua_State *L, const char *fmt, ...);

/*
  @param Image with a hole.
  @param Image's mask (white indicates a masked part).
  @param Number of patchmatch nearest neighbours search iterations to do
  @param Size of inpaint border around the masked parts. This parameter decides
         what is the width of the boundary made outwards from any masked pixel.
  @param The maximum number of pyramid levels (starting from the smallest image
         size) to inpaint. 0 means the maximum number possible.
  @param Threshold for determining which pixel is masked when downsizing the
         mask image.
*/
int lua_inpaint(lua_State *L);

/*
  This function allocates a new THByteStorage and converts the BITMAP format.
  Notice: The
*/
TorchImageStorage convert_to_torch_image(BITMAP *image);
BITMAP *convert_to_bitmap(unsigned char* im_buf, int w, int h);
/*
  The function receives 2 arguments, of two png file paths.
  A nearest neighbour searched is performed between the two images, using 2
  algorithms: PatchMatch using Patch2Vec (neural network patch comparison), and
  PatchMatch using L2 comparison.

  The function returns the average L2 distance between the NN results.
*/
static int compare_patchmatch(lua_State *L);


/*
  The function runs patch comparison using 2 methods: patch2vec and L2.
  This function receives 2 arguments, one is a path to an image file, the other
  one is the number of iterations to run. It runs the comparison against 32x32
  patches.
*/
static int compare_patchmatch_performance(lua_State *L);

static int patch2vec_image(lua_State *L);
void print_counters();

lua_State * g_L = 0;
Counters g_counters;
Counters::Counters() : n_patch_comp(0), n_nn_references(0), n_cache_nn(0) {}

extern "C" DLL_PUBLIC int luaopen_libpatchmatch2 (lua_State *L)
{
  static const luaL_Reg reg_inpaint_f[] = {
    {"nn", nn},
    {"vote", vote},
    {"inpaint", lua_inpaint},
    {"compare_nn_l2", compare_patchmatch},
    {"patch2vec_image", patch2vec_image},
    {"compare_patchmatch_performance", compare_patchmatch_performance},
    {NULL, NULL}
  };

  // out userdata methods
  static const luaL_Reg reg_inpaint_m[] = {
    {"__gc", release_bitmap}, // So lua knows how to free the object
    {NULL, NULL}
  };

  luaL_newmetatable(L, "Patchmatch.nn");
  lua_pushstring(L, "__index");
  lua_pushvalue(L, -2);
  lua_settable(L, -3);

  luaL_openlib(L, NULL, reg_inpaint_m, 0);
  luaL_openlib(L, "patchmatch", reg_inpaint_f, 0);

  g_L = L;
  return 1;
}


static int nn(lua_State *L)
{
  int nin = lua_gettop(L);
  int i = 1;

  if (nin < 2)
    {
      error(L, "Not enough arguments");
    }

  Params *p = new Params();
  RecomposeParams *rp = new RecomposeParams();
  BITMAP *borig = NULL;
  BITMAP *a = NULL, *b = NULL, *ann_prev = NULL, *ann_window = NULL, *awinsize = NULL;
  int aw = -1, ah = -1, bw = -1, bh = -1;
  VECBITMAP<unsigned char> *ab = NULL, *bb = NULL;
  VECBITMAP<float> *af = NULL, *bf = NULL;
  double *win_size = NULL;
  BITMAP *amask = NULL, *bmask = NULL;
  double scalemin = 0.5, scalemax = 2.0;  // The product of these must be one.
  int sim_mode = 0;
  int knn_chosen = -1;
  int enrich_mode = 0;
  int mode = MODE_IMAGE;
  const char * A_file_path = luaL_checkstring(L, i);  i++;
  const char * B_file_path = luaL_checkstring(L, i);  i++;

  a = load_bitmap(A_file_path);
  b = load_bitmap(B_file_path);
  aw = a->w; ah = a->h;
  bw = b->w; bh = b->h;

  if (nin >= i && luaL_checkstring(L, i))
    {
      // TODO: add more algorithms: gpucpu, cputiled, rotscale, enrich
      p->algo = ALGO_CPU;
    } i++;

  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {p->patch_w = (int)luaL_checknumber(L, i);} i++;
  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {p->nn_iters = (int)luaL_checknumber(L, i);} i++;
  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {p->rs_max = (int)luaL_checknumber(L, i);} i++;
  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {p->rs_min = (int)luaL_checknumber(L, i);} i++;
  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {p->rs_ratio = (int)luaL_checknumber(L, i);} i++;
  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {p->rs_iters = (int)luaL_checknumber(L, i);} i++;
  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {p->cores = (int)luaL_checknumber(L, i);} i++;
  if (nin >= i && !lua_isnil(L, i) && luaL_checkstring(L, i)) {amask = load_bitmap(luaL_checkstring(L, i));} i++;
  if (nin >= i && !lua_isnil(L, i) && luaL_checkstring(L, i)) {bmask = load_bitmap(luaL_checkstring(L, i));} i++;
  if (nin >= i+1 && !lua_isnil(L, i) && luaL_checknumber(L, i)) {
    p->window_w = (int)luaL_checknumber(L, i); i++;
    p->window_h = (int)luaL_checknumber(L, i);} i++;

  // TODO: complete ann_prev
  if (nin >= i && !lua_isnil(L, i) && !lua_isnil(L, i)) {error(L, "Cannot currently handle this parameter\n");} i++;

  // TODO: complete ann_window
  if (nin >= i && !lua_isnil(L, i) && !lua_isnil(L, i)) {error(L, "Cannot currently handle this parameter\n");} i++;

  // TODO: complete awinsize
  if (nin >= i && !lua_isnil(L, i) && !lua_isnil(L, i)) {error(L, "Cannot currently handle this parameter\n");} i++;

  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {
    knn_chosen = (int)luaL_checknumber(L, i);
    if (knn_chosen == 1) { knn_chosen = -1; }
    if (knn_chosen <= 0) { error(L, "knn is less than zero"); }
    if (knn_chosen > 1) { error(L, "Does not currently support knn>1");}
  } i++;

  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {
    scalemax = luaL_checknumber(L, i);
    if (scalemax <= 0) { error(L, "\nscalerange is less than zero"); }
    scalemin = 1.0/scalemax;
    if (scalemax < scalemin) {
      double temp = scalemax;
      scalemax = scalemin;
      scalemin = temp;
    }
  } i++;

  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {
    p->nn_dist = (int)luaL_checknumber(L, i);
  } i++;

  if (ann_window&&!awinsize&&!win_size) {
    error(L, "\nUsing ann_window - either awinsize or win_size should be defined.\n");
  }

  init_params(p);

  RegionMasks *amaskm = amask ? new RegionMasks(p, amask): NULL;

  BITMAP *ann = NULL; // NN field
  BITMAP *annd_final = NULL; // NN patch distance field
  BITMAP *ann_sim_final = NULL;

  VBMP *vann_sim = NULL;
  VBMP *vann = NULL;
  VBMP *vannd = NULL;

  if (mode == MODE_IMAGE) {
    // input as RGB image
    if (!a || !b) { error(L, "internal error: no a or b image"); }
    if (knn_chosen > 1) {
      p->knn = knn_chosen;
      if (sim_mode) { error(L, "rotating+scaling patches not implemented with knn (actually it is implemented it is not exposed by the wrapper)"); }
      PRINCIPAL_ANGLE *pa = NULL;
      vann_sim = NULL;
      vann = knn_init_nn(p, a, b, vann_sim, pa);
      vannd = knn_init_dist(p, a, b, vann, vann_sim);
      knn(p, a, b, vann, vann_sim, vannd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, pa);
      //      sort_knn(p, vann, vann_sim, vannd);
    } else if (sim_mode) {
      BITMAP *ann_sim = NULL;
      ann = sim_init_nn(p, a, b, ann_sim);
      BITMAP *annd = sim_init_dist(p, a, b, ann, ann_sim);
      sim_nn(p, a, b, ann, ann_sim, annd);
      if (ann_prev) { error(L, "when searching over rotations+scales, previous guess is not supported"); }
      annd_final = annd;
      ann_sim_final = ann_sim;
    } else {
      ann = init_nn(p, a, b, bmask, NULL, amaskm, 1, ann_window, awinsize);
      BITMAP *annd = init_dist(p, a, b, ann, bmask, NULL, amaskm);
      nn(p, a, b, ann, annd, amaskm, bmask, 0, 0, rp, 0, 0, 0, NULL, p->cores, ann_window, awinsize);
      if (ann_prev) minnn(p, a, b, ann, annd, ann_prev, bmask, 0, 0, rp, NULL, amaskm, p->cores);
      annd_final = annd;
    }
  }

  // Saving annd as a grayscale bmp
  {
    BITMAP *ans = create_bitmap(annd_final->w, annd_final->h);
    for (int y = 0 ; y < ah ; y++) {
      int *row = (int *)ann->line[y];
      int *ansrow = (int *)ans->line[y];
      // int *ann_row = (int *) ann->line[y];
      // int *annd_row = (int *) annd_final->line[y];
      for (int x = 0 ; x < aw ; x++) {
        int r,g,b;
        int xd = INT_TO_X(row[x]);
        int yd = INT_TO_Y(row[x]);
        double xdd = (double)255*xd/bw;
        double ydd = (double)255*yd/bh;
        r = g = b = sqrt((double)(xdd*xdd+ydd*ydd)/2);
        ansrow[x] = r|g<<8|b<<16|255<<24;
      }

    }

    // Just for demonstration, showing the target pixel distance from (0,0).
    // Looks like a heat map
    const char* ans_file_path = "/tmp/ans.bmp";
    save_bitmap(ans, ans_file_path);
    lua_pop(L, nin);
    // TODO: free memory
    push_ann_to_stack(L, ann);
    // destroy_bitmap(ans);
    // lua_pushstring(L, ans_file_path);
  }

  // clean up
  delete vann;
  delete vann_sim;
  delete vannd;
  delete p;
  delete rp;
  destroy_bitmap(a);
  destroy_bitmap(borig);
  delete ab;
  delete bb;
  delete af;
  delete bf;
  // destroy_bitmap(ann);
  destroy_bitmap(annd_final);
  destroy_bitmap(ann_sim_final);
  if (ann_prev) destroy_bitmap(ann_prev);
  if (ann_window) destroy_bitmap(ann_window);
  if (awinsize) destroy_bitmap(awinsize);

  return 1;
}

static int release_bitmap(lua_State *L)
{
  BITMAP** ud = (BITMAP**) luaL_checkudata(L, 1, "Patchmatch.nn");
  destroy_bitmap(*ud);
  *ud = 0;
  return 0;
}

int push_ann_to_stack(lua_State *L, BITMAP *ann)
{
  BITMAP **anns = (BITMAP**) lua_newuserdata(L, sizeof(BITMAP*));
  *anns = ann;
  luaL_getmetatable(L, "Patchmatch.nn");
  lua_setmetatable(L, -2);
  return 1;
}


static int vote(lua_State *L)
{
  int i = 1;
  Params *p = new Params();
  BITMAP *a = NULL, *b = NULL, *ann = NULL, *bnn = NULL;
  BITMAP *bmask = NULL, *bweight = NULL, *amask = NULL, *aweight = NULL, *ainit = NULL;
  double coherence_weight = 1, complete_weight = 1;
  int nin = lua_gettop(L);
  if (2 > nin) { error(L, "patchmatch called with < 2 input arguments"); exit(1);}
  const char * B_file_path = luaL_checkstring(L, i); i++;
  b = load_bitmap(B_file_path);
  ann = *((BITMAP**) luaL_checkudata(L, i, "Patchmatch.nn")); i++;
  if (nin >= i && !lua_isnil(L, i) && luaL_checkudata(L, i, "Patchmatch.nn")) {
    bnn = *((BITMAP**) luaL_checkudata(L, i, "Patchmatch.nn"));
  } i++;

  if (nin >= i)
    {
      if (strcmp(luaL_checkstring(L, i), "cpu") == 0)
        {p->algo = ALGO_CPU;}
      else {
          error(L, "Does not currently support other algorithms but 'cpu'\n");
      }
    } i++;

  if (nin >= i && luaL_checknumber(L, i)) {p->patch_w = (int)luaL_checknumber(L, i);} i++;
  if (nin >= i && luaL_checkstring(L, i)) {
    bmask = load_bitmap(luaL_checkstring(L, i));} i++;

  if (nin >= i && !lua_isnil(L, i))
    { error(L, "Does not currently support bweight\n"); } i++;
  if (nin >= i && !lua_isnil(L, i))
    { error(L, "Does not currently support bweight\n"); } i++;
  if (nin >= i && !lua_isnil(L, i))
    { error(L, "Does not currently support coherence_weight\n"); } i++;
  if (nin >= i && !lua_isnil(L, i))
    { error(L, "Does not currently support complete_weight\n"); } i++;
  if (nin >= i && luaL_checkstring(L, i)) {
    amask = load_bitmap(luaL_checkstring(L, i));} i++;
  if (nin >= i && !lua_isnil(L, i))
    { error(L, "Does not currently support amask\n"); } i++;
  if (nin >= i && !lua_isnil(L, i))
    { error(L, "Does not currently support aweight\n"); } i++;
  if (nin >= i && !lua_isnil(L, i))
    { error(L, "Does not currently support ainit\n"); } i++;

  RegionMasks *amaskm = amask ? new RegionMasks(p, amask): NULL;
  amask = NULL;   // amask is now managed by amaskm
  a = vote(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amaskm, aweight, ainit, NULL, NULL, 1);

  // This function also destroys amask;
  destroy_region_masks(amaskm);
  const char* ans_file_path = "/tmp/vote.bmp";
  save_bitmap(a, ans_file_path);
  lua_pop(L, nin);
  lua_pushstring(L, ans_file_path);

  delete p;
  destroy_bitmap(a);
  destroy_bitmap(ainit);
  destroy_bitmap(b);
  // These two variables are destroyed by lua
  // destroy_bitmap(ann);
  // destroy_bitmap(bnn);
  destroy_bitmap(bmask);
  destroy_bitmap(bweight);
  //  destroy_bitmap(amask);
  destroy_bitmap(aweight);
  return 1;
}


void error (lua_State *L, const char *fmt, ...) {
  va_list argp;
  va_start(argp, fmt);
  vfprintf(stderr, fmt, argp);
  va_end(argp);
    lua_close(L);
  exit(EXIT_FAILURE);
}

int lua_inpaint(lua_State *L)
{
  int nin = lua_gettop(L);
  int i = 1;

  if (nin < 2)
    {
      error(L, "Not enough arguments");
    }

  Params *p = new Params();

  const char * image_file_path = luaL_checkstring(L, i);  i++;
  const char * mask_file_path = luaL_checkstring(L, i);   i++;

  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {
    p->nn_dist = luaL_checknumber(L, i);
    if (p->nn_dist != 0) p->nn_dist = 1;
  } i++;
  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {
    p->patch_w = luaL_checknumber(L, i);
    if (p->nn_dist && (p->patch_w != 16 && p->patch_w != 32)) {
      p->patch_w = 32;   // Default value when using neural network
    }
  } i++;
  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {
    p->inpaint_add_completion_term = luaL_checknumber(L, i);
    if (p->inpaint_add_completion_term != 0) p->inpaint_add_completion_term = 1;
  } i++;
  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {
    p->inpaint_use_full_image_coherence = luaL_checknumber(L, i);
    if (p->inpaint_use_full_image_coherence != 0) p->inpaint_use_full_image_coherence = 1;
  } i++;
  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {
    p->nn_iters = luaL_checknumber(L, i);
  } i++;

  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {
    p->inpaint_border = luaL_checknumber(L, i);
  } i++;

  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {
    p->max_inpaint_levels = luaL_checknumber(L, i);
  } i++;

  if (nin >= i && !lua_isnil(L, i) && luaL_checknumber(L, i)) {
    p->mask_threshold = luaL_checknumber(L, i);
  } i++;

  BITMAP *image = load_bitmap(image_file_path);
  BITMAP *mask = load_bitmap(mask_file_path);

  init_params(p);
  p->algo = ALGO_CPU;
  inpaint(p, image, mask);

  destroy_bitmap(image);
  destroy_bitmap(mask);
  delete p;
  lua_pop(L, nin);
  return 0;
}


TorchImageStorage convert_to_torch_image(BITMAP *image)
{
  TorchImageStorage ans = {0};
  int H = image->h, W = image->w;
  ans.buffer = (unsigned char*)calloc(H*W*3, sizeof(unsigned char));

  for (int dy = 0 ; dy < H ; dy++) {
    unsigned char *abufrow = &ans.buffer[3*W*dy];
    int *imrow = ((int *) image->line[dy]);
    for (int dx = 0 ; dx < W ; dx++) {
      int ad = imrow[dx];
      unsigned char *ar = &abufrow[3*dx];
      ar[0] = ad&255;             // r
      ar[1] = (ad>>8)&255;    // g
      ar[2] = (ad>>16)&255;     // b
    }
  }
  ans.storage = THByteStorage_newWithData(ans.buffer, H*W*3);

  // Telling Torch not to free memory (we do)
  THByteStorage_clearFlag(ans.storage, TH_STORAGE_FREEMEM);
  return ans;
}


BITMAP *convert_to_bitmap(unsigned char* im_buf, int w, int h)
{
  BITMAP * ans = create_bitmap(w, h);

  for (int dy = 0 ; dy < h ; dy++)  {
    unsigned char *ims_row = &im_buf[w*dy*3];
    for (int dx = 0 ; dx < w ; dx++)  {

      unsigned char *ims_p = &ims_row[dx*3];
      int *ans_p = &((int *) ans->line[dy])[dx];
      *ans_p = ims_p[2]<<16 | ims_p[1]<<8 | ims_p[0];
    }
  }

  return ans;
}


BITMAP *downscale_image(BITMAP *image)
{
  return scale_image(image, image->h/2, image->w/2);
}


BITMAP *scale_image(BITMAP *image, int hs, int ws)
{
  unsigned char *im_scaled = 0;

  TorchImageStorage torch_image = convert_to_torch_image(image);

  lua_getglobal(g_L, "scale_image");
  luaT_pushudata(g_L, torch_image.storage, "torch.ByteStorage");
  lua_pushnumber(g_L, image->h);
  lua_pushnumber(g_L, image->w);
  lua_pushnumber(g_L, hs);
  lua_pushnumber(g_L, ws);
  lua_pushnumber(g_L, 3);

  if (lua_pcall(g_L, 6, 2, 0) != 0)
    {
      luaL_error(g_L, "error running function `f': %s",
                 lua_tostring(g_L, -1));
    }

  // Free unused memory
  free(torch_image.buffer);

  if (luaL_checknumber(g_L, -1)) {
    im_scaled = (unsigned char*)(long long)luaL_checknumber(g_L, -1);
  }
  else
    luaL_error(g_L, "Returned non torch.ByteStorage from function");

  BITMAP *ans = convert_to_bitmap(im_scaled, ws, hs);

  lua_pop(g_L, 2);
  // save_bitmap(ans, "pyramid_image_scaled.png");
  return ans;
}

static int patch2vec_image(lua_State *L)
{
  int nin = lua_gettop(L);
  int i = 1;
  if (nin < 2) {
    luaL_error(g_L, "patch2vec_image requires 2 argument2");
  }

  Params *p = new Params();
  p->patch_w = 32;
  float *tmp_vec = (float*) calloc(PATCH2VEC_LENGTH, sizeof(float));
  const char * image_file_path = luaL_checkstring(L, i);  i++;
  const char * output_file_path = luaL_checkstring(L, i);   i++;

  BITMAP *a = load_bitmap(image_file_path);

  FILE *f = fopen(output_file_path, "wb");
  if (!f) { error(L, "Does not currently support ainit\n"); }
  for (int y=0 ; y < a->h - p->patch_w+1 ; y++) {
    for (int x=0 ; x < a->w - p->patch_w+1 ; x++) {
      printf("patch2vec_image: y=%d x=%d\n", y, x);
      int retval = nn_patch2vec(a, x, y, p, tmp_vec);
      if (0 != retval) {
        memset(tmp_vec, 0, sizeof(float)*PATCH2VEC_LENGTH);
      }
      fwrite(tmp_vec, sizeof(float), PATCH2VEC_LENGTH, f);
    }
  }

  fclose(f);
  free(tmp_vec); tmp_vec = NULL;

  return 2;
}

static int compare_patchmatch_performance(lua_State *L)
{
  int nin = lua_gettop(L);
  int i = 1;
  if (nin < 2) {
    luaL_error(g_L, "compare_patchmatch_performance requires 2 argument");
  }

  const char * image_file_path = luaL_checkstring(L, i);  i++;
  int number_of_runs = luaL_checknumber(L, i); i++;


  BITMAP *a = load_bitmap(image_file_path);
  int y = 0, x = 0;
  clock_t start={0}, end={0};
  double cpu_time_used = 0;

  Params *p = new Params();
  p->patch_w = 32;
  p->nn_dist = 1;

  start = clock();
  for (int counter = number_of_runs ; counter > 0 ; counter--) {
    int retval = nn_patch_dist_ab<32>(a, x, y, a, x, y, 0x0fffffff, p);
  }
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Running %d iterations under patch2vec took %f seconds\n", number_of_runs, cpu_time_used);

  p->nn_dist = 0;
  start = clock();
  for (int counter = number_of_runs ; counter > 0 ; counter--) {
    patch_dist_ab<32, 0, 0>(p, a, x, y, a, x, y, 0x0fffffff, NULL);
  }
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Running %d iterations under L2 patchmatch took %f seconds\n", number_of_runs, cpu_time_used);

  return 2;
}


static int compare_patchmatch(lua_State *L)
{
  int nin = lua_gettop(L);
  int i = 1;

  if (nin < 2) {
    luaL_error(g_L, "compare_patchmatch requires 2 arguments");
  }

  printf("Number of threads: %d\n", omp_get_num_procs());

  const char * A_file_path = luaL_checkstring(L, i);  i++;
  const char * B_file_path = luaL_checkstring(L, i);  i++;

  char filepath[FILE_PATH_LENGTH];
  char *full_path1 = realpath(A_file_path, NULL);
  char *full_path2 = realpath(B_file_path, NULL);
  char *basename1, *basename2, dirnamebuf[FILE_PATH_LENGTH], *dir;
  char result_file_path[FILE_PATH_LENGTH];
  memcpy(dirnamebuf, full_path1, sizeof(dirnamebuf));
  dir = dirname(dirnamebuf);
  basename1 = basename(full_path1);
  basename2 = basename(full_path2);

  BITMAP *a = load_bitmap(A_file_path);
  BITMAP *b = load_bitmap(B_file_path);

  init_p2v(a);
  init_p2v(b);

  Params *p_nn = new Params();
  Params *p_l2 = new Params();
  p_l2->patch_w = p_nn->patch_w = 32;
  p_nn->nn_dist = 1;
  p_l2->nn_iters = p_nn->nn_iters = 7;

  init_params(p_nn);

  RecomposeParams *rp = new RecomposeParams();

  BITMAP *ann_nn = init_nn(p_nn, a, b, NULL, NULL, NULL, 1, NULL, NULL);
  BITMAP *annd_nn = init_dist(p_nn, a, b, ann_nn, NULL, NULL, NULL);
  nn(p_nn, a, b, ann_nn, annd_nn, NULL, NULL, 0, 0, rp, 0, 0, 0,NULL, p_nn->cores, NULL, NULL);
  minnn(p_nn, a, b, ann_nn, annd_nn, ann_nn, NULL, 0, 0, rp, NULL, NULL, p_nn->cores);

  snprintf(result_file_path, sizeof(result_file_path), "%s/%s_%s_nn.bmp", dir, basename1, basename2);
  save_bitmap(ann_nn, result_file_path);

  print_counters();

  init_params(p_l2);
  BITMAP *ann_l2 = init_nn(p_l2, a, b, NULL, NULL, NULL, 1, NULL, NULL);
  BITMAP *annd_l2 = init_dist(p_l2, a, b, ann_l2, NULL, NULL, NULL);
  nn(p_l2, a, b, ann_l2, annd_l2, NULL, NULL, 0, 0, rp, 0, 0, 0,NULL, p_l2->cores, NULL, NULL);
  minnn(p_l2, a, b, ann_l2, annd_l2, ann_l2, NULL, 0, 0, rp, NULL, NULL, p_l2->cores);

  snprintf(result_file_path, sizeof(result_file_path), "%s/%s_%s_l2.bmp", dir, basename1, basename2);
  save_bitmap(ann_l2, result_file_path);

  lua_pop(g_L, 2);
  free(full_path2);
  free(full_path1);

#ifdef USE_COUNTERS
  print_counters();
#endif // USE_COUNTERS

  return 0;
}

/*
  ret_arr should be a pointer to a float[128] buffer

  returns -1 in case of an error, 0 if the operation succeedded
*/
int nn_patch2vec(BITMAP *a, int ax, int ay, Params *p, float *ret_arr)
{
  if (NULL == ret_arr) {
    fprintf(stderr, "nn_patch2vec should be passed with valid references\n");
    exit(1);
  }

  clock_t start, end;
  double cpu_time_used;
  start = clock();

  unsigned char *abuf;
  abuf = (unsigned char*)calloc(p->patch_w*p->patch_w*3, sizeof(unsigned char));
  for (int dy = 0 ; dy < p->patch_w ; dy++) {
    unsigned char *abufrow = &abuf[p->patch_w*dy*3];
    int *arow = (int*)(a->line[ay+dy]);
    for (int dx = 0 ; dx < p->patch_w ; dx++) {
      int ad = arow[ax+dx];
      unsigned char *ar = &abufrow[3*dx];
      ar[0] = ad&255;             // r
      ar[1] = (ad>>8)&255;    // g
      ar[2] = (ad>>16)&255;     // b
    }
  }

  THByteStorage *a_th_storage, *b_th_storage;
  a_th_storage = THByteStorage_newWithData(abuf, p->patch_w*p->patch_w*3);

  lua_getglobal(g_L, "compute_patch2vec");
  luaT_pushudata(g_L, a_th_storage, "torch.ByteStorage");
  lua_pushnumber(g_L, p->patch_w);
  lua_pushnumber(g_L, p->patch_w);
  lua_pushnumber(g_L, 3);

  if (lua_pcall(g_L, 4, 2, 0) != 0)
    {
      luaL_error(g_L, "error running function `f': %s",
                 lua_tostring(g_L, -1));
      return -1;
    }

  float *vec = (float*)(long long)luaL_checknumber(g_L, -1);
  memcpy(ret_arr, vec, PATCH2VEC_LENGTH*sizeof(float));
  lua_pop(g_L, 2);

  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("patch2vec: %f sec\n", cpu_time_used);

  return 0;
}

void init_p2v(BITMAP *im)
{
  if (im->p2vd || im->p2vv) {
    return;
  }

  im->p2vd = new float [im->h*im->w*PATCH2VEC_LENGTH];
  im->p2vv = new unsigned char [im->h*im->w];

  memset(im->p2vd, 0, im->h*im->w*PATCH2VEC_LENGTH*sizeof(float));
  memset(im->p2vv, 0, im->h*im->w*sizeof(unsigned char));
}

void zero_p2v(BITMAP *im)
{
  if (im->p2vd && im->p2vv) {
    memset(im->p2vv, 0, im->h*im->w*sizeof(unsigned char));
  }
}

#ifdef USE_COUNTERS
void print_counters()
{
  printf("Number of patch comparisons: %d\n"
         "Number of neural network references requested: %d\n"
         "Number of neural network cached results returned: %d\n",
         g_counters.n_patch_comp, g_counters.n_nn_references,
         g_counters.n_cache_nn);
}
#endif // USE_COUNTERS
