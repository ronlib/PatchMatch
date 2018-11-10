#ifndef _PATCH_2_VEC_H
#define _PATCH_2_VEC_H
#include <ctime>
#include "allegro_emu.h"
#include "nn.h"
#include "lua_inpaint.h"

#include "TH/TH.h"
#include <TH/THStorage.h>
#include <TH/THTensor.h>

extern "C" {
#define LUA_LIB
#include "lua.h"
#include "lauxlib.h"
#include "luaT.h"
}

#define PATCH2VEC_LENGTH 128


template<int LENGTH>
int nn_patch_dist(int *adata, BITMAP *b, int bx, int by, int maxval, Params *p);


template<int LENGTH>
int nn_patch_dist_ab(BITMAP *a, int ax, int ay, BITMAP *b, int bx, int by, int maxval, Params *p)
{
  int adata[LENGTH*LENGTH];
  for (int dy = 0 ; dy < LENGTH ; dy++) { // copy a patch from a to adata
    int *drow = ((int *) a->line[ay+dy])+ax;
    int *adata_row = adata+(dy*LENGTH);
    for (int dx = 0 ; dx < LENGTH ; dx++) {
      adata_row[dx] = drow[dx];
    }
  }

  return nn_patch_dist<LENGTH>(adata, b, bx, by, 0, p);
}

template<int LENGTH>
int nn_patch_dist(int *adata, BITMAP *b, int bx, int by, int maxval, Params *p)
{
	if (LENGTH != p->patch_w) { fprintf(stderr, "nn_patch_dist should be called with p->patch_w==%d\n", LENGTH); exit(1); }

  clock_t start, end;
  double cpu_time_used;
  start = clock();

	unsigned char *abuf, *bbuf;
	abuf = (unsigned char*)calloc(LENGTH*LENGTH*3, sizeof(unsigned char));
	bbuf = (unsigned char*)calloc(LENGTH*LENGTH*3, sizeof(unsigned char));

	for (int dy = 0 ; dy < LENGTH ; dy++) {
    unsigned char *abufrow = &abuf[3*LENGTH*dy];
		unsigned char *bbufrow = &bbuf[3*LENGTH*dy];
    int *arow = &adata[LENGTH*dy];
    int *brow = ((int *) b->line[by+dy])+bx;
    for (int dx = 0 ; dx < LENGTH ; dx++) {
			int ad = arow[dx];
      int bd = brow[dx];
			unsigned char *ar = &abufrow[3*dx];
			unsigned char *br = &bbufrow[3*dx];
			ar[0] = ad&255;			   	// r
			ar[1] = (ad>>8)&255;		// g
			ar[2] = (ad>>16)&255;		// b
			br[0] = bd&255;
			br[1] = (bd>>8)&255;
			br[2] = (bd>>16)&255;
		}
	}

	THByteStorage *a_th_storage, *b_th_storage;
	a_th_storage = THByteStorage_newWithData(abuf, LENGTH*LENGTH*3);
	b_th_storage = THByteStorage_newWithData(bbuf, LENGTH*LENGTH*3);

	lua_getglobal(g_L, "compute_patches_distance_NN");
	luaT_pushudata(g_L, a_th_storage, "torch.ByteStorage");
	luaT_pushudata(g_L, b_th_storage, "torch.ByteStorage");
	lua_pushnumber(g_L, LENGTH);
	lua_pushnumber(g_L, LENGTH);
	lua_pushnumber(g_L, 3);

  if (lua_pcall(g_L, 5, 1, 0) != 0)
    {
      luaL_error(g_L, "error running function `f': %s",
                 lua_tostring(g_L, -1));
    }

  int lua_return_val = (int)(luaL_checknumber(g_L, -1)*INT_MAX);
  lua_pop(g_L, 1);

  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("%f sec\n", cpu_time_used);

  return lua_return_val;
}

#endif
