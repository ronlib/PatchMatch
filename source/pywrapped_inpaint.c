#include "defineall.h"

double* G_globalSimilarity = NULL;
int G_initSim = 0;

void mem_alloc(int ** out_buf, int* allocated_int_length)
{
	*out_buf = malloc(100*sizeof(int));
	printf("%p\n", *out_buf);
	*allocated_int_length = 100;
}

double max1(double a, double b)
{
	return (a + b + fabs(a-b) ) / 2;
}

double min1(double a, double b)
{
	return (a + b - fabs(a-b) ) / 2;
}


int pyinpaint(unsigned char* img_data, int nchannels, unsigned char* in_mask, int H, int W, unsigned char ** out_buf)
{

	int channels=nchannels;
	int ** mask;
	/* CvMat mat = cvMat(H, W, CV_8UC3, img_data); */
	IplImage* input_img = cvCreateImageHeader(cvSize(W, H), IPL_DEPTH_8U, 3);
	IplImage* input_mask = cvCreateImageHeader(cvSize(W, H), IPL_DEPTH_8U, 1);
	IplImage * output_img;
	printf("cvSetData\n");
	/* img = cvCreateMatHeader(H, W, CV_8UC3); */
	cvSetData(input_img, img_data, input_img->width*input_img->nChannels);
	cvSetData(input_mask, in_mask, input_mask->width*input_mask->nChannels);

	/* printf("Showing image\n"); */
	/* cvShowImage("bla", (const void *)input_img); */
	/* cvWaitKey(10000); */


	/* printf("Showing mask\n"); */
	/* cvShowImage("Mask", (const void *)input_mask); */
	/* cvWaitKey(10000); */

	mask = (int **) calloc((int)H,sizeof(int*));
	for ( int i=0 ; i<H ; i++)
		mask[i] = (int *) calloc((int)W,sizeof(int));

	for ( int i = 0 ; i < H ; ++i )
		for ( int j = 0 ; j < W ; ++j )
			if ( in_mask[i*W+j]>0 )
				mask[i][j]=1;

	Inpaint_P inp = initInpaint();
	input_img->width = W;
	input_img->height = H;
	output_img = inpaint(inp, (IplImage*)input_img, (int**)mask, 2);
	*out_buf = output_img->imageData;



	return 0;
}

/*
int main()
{
	const unsigned char * data = (const unsigned char*){100, 100, 100, 200, 200, 200, 100, 100, 100, 200, 200, 200};
	CvMat* img;


	img = cvLoadImageM("./images/forest.bmp", CV_LOAD_IMAGE_COLOR);
	cvShowImage("bla", (const void *)img);

	cvWaitKey(10000);
	pyinpaint(data, 3, (unsigned char*)0, 2, 2);

	return 0;
}
*/
