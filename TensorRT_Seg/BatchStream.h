#pragma once

#ifndef BATCH_STREAM_H
#define BATCH_STREAM_H

#include <vector>
#include <assert.h>
#include <algorithm>
#include "NvInfer.h"


class batchstream
{
public:
	batchstream(int maxbatch)
		:Maxbatch(maxbatch)
	{
		int mImageSize = 1 * 3 * 512 * 512;
		im.data = 0;
		im.h = 512;
		im.w = 512;
		im.c = 3;
		im.data = (float*)calloc(mImageSize, sizeof(float));
	}
	bool next()
	{
		if (Count == Maxbatch) return false;
		std::string names = std::string("D:\\pytorch\\light-weight-refinenet\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\batch500test\\batch") + std::to_string(Count++) + std::string(".jpg");
		IplImage* src = cvLoadImage(names.c_str());
		unsigned char *data = (unsigned char *)src->imageData;
		int h = src->height;
		int w = src->width;
		int c = src->nChannels;
		int step = src->widthStep;//排列的图像行大小，以字节为单位
		//im = make_image(w, h, c);/*做一个长宽对等的空image容器*/
		int i, j, k, count = 0;;
		//像素存储方式由[行-列-通道]转换为[通道-行-列] hwc  --->chw
		for (k = 0; k < c; ++k) {
			for (i = 0; i < h; ++i) {
				for (j = 0; j < w; ++j) {
					im.data[count++] = data[i*step + j * c + k] / 255.;   //chw
				}
			}
		}
		std::cout << Count << std::endl;
		rgb_image(im);
		cvReleaseImage(&src);
		return true;
	}
	float* get_image()
	{
		return im.data;
	}
	void rgb_image(image im);
	int getBatchSize() const { return mBatchSize; }
private:
	int Maxbatch;
	int mBatchSize = 1;
	int Count = 0;
	image im;

};




void batchstream::rgb_image(image im)
{
	int i;
	for (i = 0; i < im.w*im.h; ++i) {
		float swap = im.data[i];
		im.data[i] = im.data[i + im.w*im.h * 2];
		im.data[i + im.w*im.h * 2] = swap;
	}
}



#endif

