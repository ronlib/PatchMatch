/* #include "THGeneral.h" */
/* #include "THAtomic.h" */
/* #define TH_GENERIC_FILE */
/* #include "THStorage.h" */
/* #include "generic/THTensor.h" */
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
/* #include "auxiliarydefs.h" */


static const char *LIBRARY_NAME = "inpaint";
static int lua_inpaint (lua_State *L);
static int l_tensorsize(lua_State *L);
static int l_inpaint_str(lua_State *L);

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

LUALIB_API int luaopen_luainpaint (lua_State *L) {
	static const luaL_Reg reg_inpaint[] = {
		{"inpaint",   lua_inpaint},
		{"tensorsize", l_tensorsize},
		{"inpaint2", l_inpaint_str},
		{NULL, NULL}
	};

	/* luaL_register(L, (void*)0, LIBRARY_NAME); */
  luaL_register(L, LIBRARY_NAME, reg_inpaint);
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

static int l_inpaint_str(lua_State *L) {
	IplImage *image, *mask;
	CvMat * cloned_mask, *cloned_image, tmp_mat;
	unsigned char * out_buf = 0;
	const char * output_file_name = "/tmp/result.bmp";
	int result = 0;
	const char * image_file_path = luaL_checkstring(L, 1);
	const char * mask_file_path = luaL_checkstring(L, 2);
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

	/* cvShowImage("Mask", (const void *)mask); */
	/* cvWaitKey(10000); */

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
