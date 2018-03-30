/* #include "THGeneral.h" */
/* #include "THAtomic.h" */
/* #define TH_GENERIC_FILE */
/* #include "THStorage.h" */
/* #include "generic/THTensor.h" */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include "TH/TH.h"
#include <TH/THStorage.h>
#include <TH/THTensor.h>
#include <cv.h>
#include <highgui.h>

#include <math.h>
#define LUA_LIB
#include "lua.h"
#include "lauxlib.h"
#include "luaT.h"
#include "wrapped_inpaint.h"
#include "lua_inpaint.h"
/* #include "auxiliarydefs.h" */


static const char *LIBRARY_NAME = "inpaint";
static int lua_inpaint (lua_State *L);
static int l_tensorsize(lua_State *L);
static int l_inpaint_str(lua_State *L);

static lua_State * g_L = 0;

/*
double* G_globalSimilarity = NULL;
int G_initSim = 0;

double max1(double a, double b)
{
	return (a + b + fabs(a-b) ) / 2;
}

double min1(double a, double b)
{
	return (a + b - fabs(a-b) ) / 2;
}
*/

void error (lua_State *L, const char *fmt, ...) {
	va_list argp;
	va_start(argp, fmt);
	vfprintf(stderr, fmt, argp);
	va_end(argp);
	lua_close(L);
	exit(EXIT_FAILURE);
}

#ifdef _MSC_VER
__declspec(dllexport) LUALIB_API int luaopen_luainpaint (lua_State *L)
#else
LUALIB_API int luaopen_luainpaint (lua_State *L)
#endif
{
	static const luaL_Reg reg_inpaint[] = {
		/* {"inpaint2",   lua_inpaint}, */
		/* {"tensorsize", l_tensorsize}, */
		{"inpaint", l_inpaint_str},
		{NULL, NULL}
	};

  luaL_register(L, LIBRARY_NAME, reg_inpaint);
	if (luaL_loadfile(L, "patch2vec.lua") || lua_pcall(L, 0, 0, 0))
		{
			error(L, "cannot run file: %s",
						lua_tostring(L, -1));
		}
  return 1;
}

static int lua_inpaint (lua_State *L) {
	lua_pushnumber(L, sin(luaL_checknumber(L, 1)));
	return 1;
}

static int l_tensorsize(lua_State *L) {
	int size;
  /* float *num; */
	THFloatTensor *a = (THFloatTensor *)luaT_toudata(L, 1, "torch.FloatTensor");
	/* top = lua_gettop(L); */

	size = THFloatTensor_size(a, 0);
	lua_pop(L, 1);
	lua_pushnumber(L, (double)size);
	/* printf("Size: %d\n", THFloatTensor_size(a, 0)); */
	/* num = (float*)THFloatTensor_data(a); */
	/* printf("num = %f\n", num[1]); */
	return 1;
}

static int l_inpaint(lua_State *L) {
	int size;
	THFloatTensor *a = (THFloatTensor *)luaT_toudata(L, 1, "torch.FloatTensor");

	size = THFloatTensor_size(a, 0);
	lua_pop(L, 1);
	lua_pushnumber(L, (double)size);

	return 1;
}

int g_nn_count = 0;

static int l_inpaint_str(lua_State *L) {
	IplImage *image, *mask;
	CvMat * cloned_mask, *cloned_image, tmp_mat;
	unsigned char * out_buf = 0;
	const char * output_file_name = "/tmp/result.bmp";
	int result = 0;
	const char * image_file_path = luaL_checkstring(L, 1);
	const char * mask_file_path = luaL_checkstring(L, 2);
	g_L = L;
	++g_nn_count;
	image = cvLoadImage(image_file_path, 1);
	cvCvtColor(image, image, CV_BGR2RGB);
	/* cvReleaseImage(&image); */
	/* image = 0; */
	mask = cvLoadImage(mask_file_path, CV_LOAD_IMAGE_GRAYSCALE);
	image_file_path = 0;
	mask_file_path = 0;
	lua_pop(L, 2);

	// Cloning the images as Mat in to remove 4 byte padding, which inpaint() does not support
	cvGetMat(mask, &tmp_mat, NULL, 0);
	cloned_mask = cvCloneMat(&tmp_mat);
	cvReleaseImage(&mask);
	mask = 0;

	cvGetMat(image, &tmp_mat, NULL, 0);
	cloned_image = cvCloneMat(&tmp_mat);
	cvReleaseImage(&image);
	image = 0;

	//////////////////////
	/* { */
	/* 	unsigned char* tmp_buf = 0; */
	/* 	CvMat tmp_mat, *cloned_source, *cloned_target; */
	/* 	double lua_return_val = 0; */
	/* 	THByteStorage *image_th_storage, *mask_th_storage; */

	/* 	tmp_buf = calloc(cloned_image->rows*cloned_image->cols*3, sizeof(unsigned char)); */
	/* 	memcpy(tmp_buf, cloned_image->data.ptr, cloned_image->rows*cloned_image->cols*3); */
	/* 	image_th_storage = THByteStorage_newWithData(tmp_buf, cloned_image->rows*cloned_image->cols*3); */

	/* 	tmp_buf = calloc(cloned_mask->rows*cloned_mask->cols*1, sizeof(unsigned char)); */
	/* 	mask_th_storage = THByteStorage_newWithData(tmp_buf, cloned_mask->rows*cloned_mask->cols*1); */

	/* 	lua_getglobal(g_L, "called_from_c"); */
	/* 	luaT_pushudata(g_L, image_th_storage, "torch.ByteStorage"); */
	/* 	luaT_pushudata(g_L, mask_th_storage, "torch.ByteStorage"); */
	/* 	lua_pushnumber(g_L, cloned_image->rows); */
	/* 	lua_pushnumber(g_L, cloned_image->cols); */

	/* 	if (lua_pcall(g_L, 4, 1, 0) != 0) */
	/* 		{ */
	/* 			luaL_error(g_L, "error running function `f': %s", */
	/* 								 lua_tostring(g_L, -1)); */
	/* 		} */

	/* 	lua_return_val = luaL_checknumber(g_L, -1); */
	/* 	lua_pop(g_L, 1); */
	/* } */
	/* return 0; */


//////////////////////////////

	result = wrapped_inpaint(cloned_image->data.ptr, 3, cloned_mask->data.ptr, cloned_image->rows, cloned_image->cols, &out_buf);

	// Building an image from the output buffer
	image = cvCreateImageHeader(cvSize(cloned_image->cols, cloned_image->rows), IPL_DEPTH_8U, 3);

	cvSetData(image, out_buf, cloned_image->cols*3);
	cvCvtColor(image, image, CV_RGB2BGR);

	// Cleaning up memory
	cvReleaseMat(&cloned_mask);
	cvReleaseMat(&cloned_image);
	cloned_mask = cloned_mask = 0;

	// Saving the output image as a file
	cvSaveImage(output_file_name, image, 0);
	cvReleaseImageHeader(&image);
	image = 0;

	free(out_buf);

	lua_pushstring(L, output_file_name);
	return 1;


}

#ifdef DPNN_PATCH_DISTANCE

int distanceDPNNMaskedImage(MaskedImage_P source,int xs,int ys, MaskedImage_P target,int xt,int yt, int S)
{
	CvMat tmp_mat, *cloned_source, *cloned_target;
	int xss = xs-S, xes = xs+S, yss = ys-S, yes = ys+S;
	int xst = xt-S, xet = xt+S, yst = yt-S, yet = yt+S;
	double lua_return_val = 0;
	THByteStorage *source_th_storage, *target_th_storage;
	unsigned char* tmp_buf = 0;

	if ((xss < 1 || xst < 1 || yss < 1 || yst < 1) ||
			(xes >= source->image->height-1 || yes >= source->image->width-1) ||
			(xet >= target->image->height-1 || yet >= target->image->width-1))
		{
			return DSCALE;
		}

	/* cvGetMat(source->image, &tmp_mat, NULL, 0); */
	cvGetSubRect(source->image, &tmp_mat, cvRect(yss, xss, 2*S, 2*S));
	cloned_source = cvCloneMat(&tmp_mat);
	cvGetSubRect(target->image, &tmp_mat, cvRect(yst, xst, 2*S, 2*S));
	cloned_target = cvCloneMat(&tmp_mat);

	tmp_buf = calloc(cloned_source->rows*cloned_source->cols*3, sizeof(unsigned char));
	memcpy(tmp_buf, cloned_source->data.ptr, cloned_source->rows*cloned_source->cols*3);
	source_th_storage = THByteStorage_newWithData(tmp_buf, cloned_source->rows*cloned_source->cols*3);

	tmp_buf = calloc(cloned_target->rows*cloned_target->cols*3, sizeof(unsigned char));
	memcpy(tmp_buf, cloned_target->data.ptr, cloned_target->rows*cloned_target->cols*3);
	target_th_storage = THByteStorage_newWithData(tmp_buf, cloned_target->rows*cloned_target->cols*3);
	lua_getglobal(g_L, "compute_patches_distance_NN");
	luaT_pushudata(g_L, source_th_storage, "torch.ByteStorage");
	luaT_pushudata(g_L, target_th_storage, "torch.ByteStorage");
	lua_pushnumber(g_L, cloned_source->rows);
	lua_pushnumber(g_L, cloned_source->cols);
	lua_pushnumber(g_L, 3);

	if (lua_pcall(g_L, 5, 1, 0) != 0)
		{
			luaL_error(g_L, "error running function `f': %s",
                 lua_tostring(g_L, -1));
		}

	lua_return_val = luaL_checknumber(g_L, -1);
	lua_pop(g_L, 1);
	printf("Distance: %f\n", lua_return_val*DSCALE);

	return lua_return_val*DSCALE;

}

#endif
