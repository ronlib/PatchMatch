#include "defineall.h"

int pyinpaint(unsigned char* img, int nchannels, unsigned char* in_mask, int H, int W)
{

	int channels=nchannels;
	int ** mask;
	CvMat mat = cvMat(H, W, CV_8UC3, img);
	CvMat* mat2 = cvCreateMatHeader(H, W, CV_8UC3);
	cvSetData(mat2, img, mat2->cols*nchannels);
	cvShowImage("bla", (const void *)&mat);
	cvWaitKey(10000);
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
