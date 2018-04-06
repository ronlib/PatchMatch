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
	const char * image_file_path = luaL_checkstring(L, i);	i++;
	const char * mask_file_path = luaL_checkstring(L, i);	i++;

	BITMAP *image = load_bitmap(image_file_path);
	BITMAP *mask = load_bitmap(mask_file_path);

	if (nin >= i && luaL_checkstring(L, i))
		{
			// TODO: add more algorithms
			p->algo = ALGO_CPU;
		} i++;

	if (nin >= i && luaL_checknumber(L, i)) {p->patch_w = (int)luaL_checknumber(L, i);} i++;
	if (nin >= i && luaL_checknumber(L, i)) {p->nn_iters = (int)luaL_checknumber(L, i);} i++;
	if (nin >= i && luaL_checknumber(L, i)) {p->rs_max = (int)luaL_checknumber(L, i);} i++;
	if (nin >= i && luaL_checknumber(L, i)) {p->rs_min = (int)luaL_checknumber(L, i);} i++;
	if (nin >= i && luaL_checknumber(L, i)) {p->rs_ratio = (int)luaL_checknumber(L, i);} i++;
	if (nin >= i && luaL_checknumber(L, i)) {p->rs_iters = (int)luaL_checknumber(L, i);} i++;
	if (nin >= i && luaL_checknumber(L, i)) {p->cores = (int)luaL_checknumber(L, i);} i++;
}


void error (lua_State *L, const char *fmt, ...) {
	va_list argp;
	va_start(argp, fmt);
	vfprintf(stderr, fmt, argp);
	va_end(argp);
	lua_close(L);
	exit(EXIT_FAILURE);
}
