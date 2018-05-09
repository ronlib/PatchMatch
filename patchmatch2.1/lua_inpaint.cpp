#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

#define MODE_IMAGE  0
#define MODE_VECB   1
#define MODE_VECF   2

#if defined _WIN32 || defined __CYGWIN__
  #ifdef BUILDING_DLL
    #ifdef __GNUC__
      #define DLL_PUBLIC __attribute__ ((dllexport))
    #else
      #define DLL_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define DLL_PUBLIC __attribute__ ((dllimport))
    #else
      #define DLL_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define DLL_LOCAL
#else
  #if __GNUC__ >= 4
    #define DLL_PUBLIC __attribute__ ((visibility ("default")))
    #define DLL_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define DLL_PUBLIC
    #define DLL_LOCAL
  #endif
#endif

BITMAP *load_bitmap(const char *filename);
void save_bitmap(BITMAP *bmp, const char *filename);
static int nn(lua_State *L);
static int vote(lua_State *L);
static int release_bitmap(lua_State *L);
int push_ann_to_stack(lua_State *L, BITMAP *ann);
void error (lua_State *L, const char *fmt, ...);

static lua_State * g_L = 0;

#ifdef _MSC_VER
__declspec(dllexport) LUALIB_API int luaopen_luainpaint (lua_State *L)
#else
	extern "C" DLL_PUBLIC int luaopen_libpatchmatch2 (lua_State *L)
#endif
{
	static const luaL_Reg reg_inpaint_f[] = {
		{"nn", nn},
		{"vote", vote},
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
	const char * A_file_path = luaL_checkstring(L, i);	i++;
	const char * B_file_path = luaL_checkstring(L, i);	i++;

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
	a = vote(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amaskm, aweight, ainit, NULL, NULL, 1);

	destroy_region_masks(amaskm);
	const char* ans_file_path = "/tmp/vote.bmp";
	save_bitmap(a, ans_file_path);
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


int nn16_patch_dist(int *adata, BITMAP *b, int bx, int by, int maxval, Params *p)
{
	if (16 != p->patch_w) { fprintf(stderr, "nn16_patch_dist should be called with p->patch_w==16\n"); exit(1); }

	unsigned char *abuf, *bbuf;
	abuf = (unsigned char*)calloc(16*16*3, sizeof(unsigned char));
	bbuf = (unsigned char*)calloc(16*16*3, sizeof(unsigned char));

	for (int dy = 0 ; dy < 16 ; dy++) {
		unsigned char *arow = abuf + dy*16*3;
		int *brow = ((int *) b->line[by+dy])+bx;
		for (int dx = 0 ; dx < 16 ; dx++) {
			int ad = adata[dx];
			int bd = brow[dx];
			unsigned char *ar = &abuf[3*dx];
			unsigned char *br = &bbuf[3*dx];
			ar[0] = ad&255;			   	// r
			ar[1] = (ad>>8)&&255;		// g
			ar[2] = (ad>>16)&255;		// b
			br[0] = bd&255;
			br[1] = (bd>>8)&&255;
			br[2] = (bd>>16)&255;
		}
	}

	THByteStorage *a_th_storage, *b_th_storage;
	a_th_storage = THByteStorage_newWithData(abuf, 16*16*3);
	b_th_storage = THByteStorage_newWithData(bbuf, 16*16*3);

	lua_getglobal(g_L, "compute_patches_distance_NN");
	luaT_pushudata(g_L, a_th_storage, "torch.ByteStorage");
	luaT_pushudata(g_L, b_th_storage, "torch.ByteStorage");
	lua_pushnumber(g_L, 16);
	lua_pushnumber(g_L, 16);
	lua_pushnumber(g_L, 3);

	if (lua_pcall(g_L, 5, 1, 0) != 0)
		{
			luaL_error(g_L, "error running function `f': %s",
                 lua_tostring(g_L, -1));
		}

	int lua_return_val = (int)luaL_checknumber(g_L, -1);
	lua_pop(g_L, 1);

	return 0;
}
