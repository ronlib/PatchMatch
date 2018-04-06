#include "lua.h"
#include "lauxlib.h"
#include "luaT.h"
#include "vecnn.h"
#include "allegro_emu.h"

BITMAP *load_bitmap(const char *filename);
static int patchmatch(lua_State *L);
void error (lua_State *L, const char *fmt, ...);

#ifdef _MSC_VER
__declspec(dllexport) LUALIB_API int luaopen_luainpaint (lua_State *L)
#else
LUALIB_API int luaopen_luainpaint (lua_State *L)
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
	BITMAP *a = NULL, *b = NULL, *ann_prev = NULL, *ann_window = NULL, *awinsize = NULL;
  double *win_size = NULL;
  BITMAP *amask = NULL, *bmask = NULL;
  double scalemin = 0.5, scalemax = 2.0;  // The product of these must be one.
	int sim_mode = 0;
	int knn_chosen = -1;
	int enrich_mode = 0;
	const char * image_file_path = luaL_checkstring(L, i);	i++;
	const char * mask_file_path = luaL_checkstring(L, i);	i++;

	BITMAP *image = load_bitmap(image_file_path);
	BITMAP *mask = load_bitmap(mask_file_path);

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
}


void error (lua_State *L, const char *fmt, ...) {
	va_list argp;
	va_start(argp, fmt);
	vfprintf(stderr, fmt, argp);
	va_end(argp);
	lua_close(L);
	exit(EXIT_FAILURE);
}
