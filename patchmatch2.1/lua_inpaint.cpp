#include <math.h>
extern "C" {
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
static int patchmatch(lua_State *L);
void error (lua_State *L, const char *fmt, ...);

#ifdef _MSC_VER
__declspec(dllexport) LUALIB_API int luaopen_luainpaint (lua_State *L)
#else
	extern "C" DLL_PUBLIC int luaopen_libpatchmatch2 (lua_State *L)
#endif
{
	static const luaL_Reg reg_inpaint[] = {
		{"inpaint", patchmatch},
		{NULL, NULL}
	};
  luaL_register(L, "patchmatch", reg_inpaint);
	return 1;
}


static int patchmatch(lua_State *L)
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

	if (nin >= i && luaL_checkstring(L, i))
		{
			// TODO: add more algorithms: gpucpu, cputiled, rotscale, enrich
			p->algo = ALGO_CPU;
		} i++;

	if (nin >= i && luaL_checknumber(L, i)) {p->patch_w = (int)luaL_checknumber(L, i);} i++;
	if (nin >= i && luaL_checknumber(L, i)) {p->nn_iters = (int)luaL_checknumber(L, i);} i++;
	if (nin >= i && luaL_checknumber(L, i)) {p->rs_max = (int)luaL_checknumber(L, i);} i++;
	if (nin >= i && luaL_checknumber(L, i)) {p->rs_min = (int)luaL_checknumber(L, i);} i++;
	if (nin >= i && luaL_checknumber(L, i)) {p->rs_ratio = (int)luaL_checknumber(L, i);} i++;
	if (nin >= i && luaL_checknumber(L, i)) {p->rs_iters = (int)luaL_checknumber(L, i);} i++;
	if (nin >= i && luaL_checknumber(L, i)) {p->cores = (int)luaL_checknumber(L, i);} i++;
	if (nin >= i+1 && luaL_checknumber(L, i)) {
		p->window_w = (int)luaL_checknumber(L, i); i++;
		p->window_h = (int)luaL_checknumber(L, i);} i++;

	// TODO: complete ann_prev
	if (nin >= i && lua_isnil(L, i)) {} else {error(L, "Cannot currently handle this parameter");} i++;

	// TODO: complete ann_window
	if (nin >= i && lua_isnil(L, i)) {} else {error(L, "Cannot currently handle this parameter");} i++;

	// TODO: complete awinsize
	if (nin >= i && lua_isnil(L, i)) {} else {error(L, "Cannot currently handle this parameter");} i++;

	if (nin >= i && luaL_checknumber(L, i)) {
		knn_chosen = (int)luaL_checknumber(L, i);
		if (knn_chosen == 1) { knn_chosen = -1; }
		if (knn_chosen <= 0) { error(L, "knn is less than zero"); }
		if (knn_chosen > 1) { error(L, "Does not currently support knn>1");}
	} i++;

	if (nin >= i && luaL_checknumber(L, i)) {
		scalemax = luaL_checknumber(L, i);
		if (scalemax <= 0) { error(L, "\nscalerange is less than zero"); }
    scalemin = 1.0/scalemax;
    if (scalemax < scalemin) {
      double temp = scalemax;
      scalemax = scalemin;
      scalemin = temp;
    }
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
			int *row = (int *)annd_final->line[y];
			int *arow = (int *)ans->line[y];
			for (int x = 0 ; x < aw ; x++) {
				int r,g,b;
				int xd = INT_TO_X(row[x]);
				int yd = INT_TO_Y(row[x]);
				r = g = b = sqrt((float)(xd*xd+yd*yd)/2);
				arow[x] = int(r*255)|(int(g*255)<<8)|(int(b*255)<<16);
			}

		}

		const char* ans_file_path = "/tmp/ans.bmp";
		save_bitmap(ans, ans_file_path);
		destroy_bitmap(ans);
		lua_pushstring(L, ans_file_path);
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
  destroy_bitmap(ann);
  destroy_bitmap(annd_final);
  destroy_bitmap(ann_sim_final);
  if (ann_prev) destroy_bitmap(ann_prev);
  if (ann_window) destroy_bitmap(ann_window);
  if (awinsize) destroy_bitmap(awinsize);

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
