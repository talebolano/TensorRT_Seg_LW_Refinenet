#include <stdio.h>
#include <math.h>

#include "image.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/types_c.h"
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio_c.h"
#include "opencv2/imgcodecs/imgcodecs_c.h"
#endif
#endif


int windows = 0;
static const float kMean[3] = { 0.485f, 0.456f, 0.406f };
static const float kStdDev[3] = { 0.229f, 0.224f, 0.225f };
static const float map[7][3] = { {0,0,0} ,
								{0.5,0,0},
								{0,0.5,0},
								{0,0,0.5},
								{0.5,0.5,0},
								{0.5,0,0.5},
								{0,0.5,0} };

void free_image(image m)
{
	if (m.data) {
		free(m.data);
	}
}



image make_empty_image(int w, int h, int c)
{
	image out;
	out.data = 0;
	out.h = h;
	out.w = w;
	out.c = c;
	return out;
}


image make_image(int w, int h, int c)
{
	image out = make_empty_image(w, h, c);
	out.data = (float*)calloc(h*w*c, sizeof(float));
	return out;
}

image ipl_to_image(IplImage* src)
{
	unsigned char *data = (unsigned char *)src->imageData;
	int h = src->height;
	int w = src->width;
	int c = src->nChannels;
	int step = src->widthStep;//排列的图像行大小，以字节为单位
	image out = make_image(w, h, c);/*做一个长宽对等的空image容器*/
	int i, j, k, count = 0;;
	//像素存储方式由[行-列-通道]转换为[通道-行-列] hwc  --->chw
	for (k = 0; k < c; ++k) {
		for (i = 0; i < h; ++i) {
			for (j = 0; j < w; ++j) {
				out.data[count++] = data[i*step + j * c + k] / 255.;
			}
		}
	}
	return out;
}

void rgbgr_image(image im)
{
	int i;
	for (i = 0; i < im.w*im.h; ++i) {
		float swap = im.data[i];
		im.data[i] = im.data[i + im.w*im.h * 2];
		im.data[i + im.w*im.h * 2] = swap;
	}
}


image copy_image(image p)
{
	image copy = p;
	copy.data = (float*)calloc(p.h*p.w*p.c, sizeof(float));
	memcpy(copy.data, p.data, p.h*p.w*p.c * sizeof(float));
	return copy;
}

static float get_pixel(image m, int x, int y, int c)
{
	assert(x < m.w && y < m.h && c < m.c);
	return m.data[c*m.h*m.w + y * m.w + x];
}


void constrain_image(image im)
{
	int i;
	for (i = 0; i < im.w*im.h*im.c; ++i) {
		if (im.data[i] < 0) im.data[i] = 0;
		if (im.data[i] > 1) im.data[i] = 1;
	}
}

void show_image(image p, const char *name)
{
	int x, y, k;
	image copy = copy_image(p);
	constrain_image(copy);
	if (p.c == 3) rgbgr_image(copy);    // rgb -->brg
	//normalize_image(copy);
	char buff[256];
	//sprintf(buff, "%s (%d)", name, windows);
	sprintf(buff, "%s", name);
	IplImage *disp = cvCreateImage(cvSize(p.w, p.h), IPL_DEPTH_8U, p.c);
	int step = disp->widthStep;
	cvNamedWindow(buff, CV_WINDOW_NORMAL);
	//cvMoveWindow(buff, 100*(windows%10) + 200*(windows/10), 100*(windows%10));
	++windows;
	for (y = 0; y < p.h; ++y) {
		for (x = 0; x < p.w; ++x) {
			for (k = 0; k < p.c; ++k) {
				disp->imageData[y*step + x * p.c + k] = (unsigned char)(get_pixel(copy, x, y, k) * 255);
			}
		}
	}
	free_image(copy);
	if (0) {
		int w = 448;
		int h = w * p.h / p.w;
		if (h > 1000) {
			h = 1000;
			w = h * p.w / p.h;
		}
		IplImage *buffer = disp;
		disp = cvCreateImage(cvSize(w, h), buffer->depth, buffer->nChannels);
		cvResize(buffer, disp, CV_INTER_LINEAR);
		cvReleaseImage(&buffer);
	}

	cvShowImage(buff, disp);
	cvReleaseImage(&disp);

}

static float get_pixel_extend(image m, int x, int y, int c)
{
	if (x < 0 || x >= m.w || y < 0 || y >= m.h) return 0;
	/*
	if(x < 0) x = 0;
	if(x >= m.w) x = m.w-1;
	if(y < 0) y = 0;
	if(y >= m.h) y = m.h-1;
	*/
	if (c < 0 || c >= m.c) return 0;
	return get_pixel(m, x, y, c);
}

static void set_pixel(image m, int x, int y, int c, float val)
{
	if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
	assert(x < m.w && y < m.h && c < m.c);
	m.data[c*m.h*m.w + y * m.w + x] = val;
}
static void add_pixel(image m, int x, int y, int c, float val)
{
	assert(x < m.w && y < m.h && c < m.c);
	m.data[c*m.h*m.w + y * m.w + x] += val;
}

image normal_image(image im)
{
	for (int c = 0; c < im.c; c++)
	{
		for (int h = 0; h < im.h; h++)
		{
			for (int w = 0; w < im.w; w++)
			{
				int dstIdx = c * im.h * im.w + h * im.w + w;
				im.data[dstIdx] = (im.data[dstIdx] - kMean[c]) / kStdDev[c];

			}
		}
	}
	return im;
}

void flip_image(image a)
{
	int i, j, k;
	for (k = 0; k < a.c; ++k) {
		for (i = 0; i < a.h; ++i) {
			for (j = 0; j < a.w / 2; ++j) {
				int index = j + a.w*(i + a.h*(k));
				int flip = (a.w - j - 1) + a.w*(i + a.h*(k));
				float swap = a.data[flip];
				a.data[flip] = a.data[index];
				a.data[index] = swap;
			}
		}
	}
}

image resize_image(image im, int w, int h)
{
	//chw
	image resized = make_image(w, h, im.c);//要变形的长宽
	image part = make_image(w, im.h, im.c);//原来的长，要变的宽
	int r, c, k;
	float w_scale = (float)(im.w - 1) / (w - 1);//原来的/要变的  if <1
	float h_scale = (float)(im.h - 1) / (h - 1);
	for (k = 0; k < im.c; ++k) {
		for (r = 0; r < im.h; ++r) {
			for (c = 0; c < w; ++c) {
				float val = 0;
				if (c == w - 1 || im.w == 1) {
					val = get_pixel(im, im.w - 1, r, k);
				}
				else {
					float sx = c * w_scale;
					int ix = (int)sx;
					float dx = sx - ix;
					//nearest
					val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix + 1, r, k);
				}
				set_pixel(part, c, r, k, val);
			}
		}
	}
	for (k = 0; k < im.c; ++k) {
		for (r = 0; r < h; ++r) {
			float sy = r * h_scale;
			int iy = (int)sy;
			float dy = sy - iy;
			for (c = 0; c < w; ++c) {
				float val = (1 - dy) * get_pixel(part, c, iy, k);
				set_pixel(resized, c, r, k, val);
			}
			if (r == h - 1 || im.h == 1) continue;
			for (c = 0; c < w; ++c) {
				float val = dy * get_pixel(part, c, iy + 1, k);
				add_pixel(resized, c, r, k, val);
			}
		}
	}

	free_image(part);
	return resized;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
	int ix = (int)floorf(x);
	int iy = (int)floorf(y);

	float dx = x - ix;
	float dy = y - iy;

	float val = (1 - dy) * (1 - dx) * get_pixel_extend(im, ix, iy, c) +
		dy * (1 - dx) * get_pixel_extend(im, ix, iy + 1, c) +
		(1 - dy) *   dx   * get_pixel_extend(im, ix + 1, iy, c) +
		dy * dx   * get_pixel_extend(im, ix + 1, iy + 1, c);
	return val;
}


int seven_ind_max(float *a)
{
	int index = 0;
	float swap;

	for (int i = 0; i < 7; ++i)
	{
		if (a[0] < a[i])
		{
			swap = a[0];
			a[0] = a[i];
			a[i] = swap;
			index = i;
		}
	}
	return index;
}

//用nearest 【7，128，128】-->【7，512，512】-->【3，512，512】
image Tranpose(float *prob)
{
	int h = 128;
	int w = 128;
	int c = 7;
	image out = make_image(w, h, c);
	for (int y = 0; y < c; ++y) {
		for (int x = 0; x < h; ++x) {
			for (int k = 0; k < w; ++k)
			{
				out.data[y * 128 * 128 + x * 128 + k] = prob[y * 128 * 128 + x * 128 + k];
			}
		}
	}
	//chw
	image in_s = resize_image(out, 512, 512);//[7,512,512]
	free_image(out);
	int index;
	image real_out = make_image(512, 512, 3);
	float x[7];
	for (int ih = 0; ih < 512; ++ih) {
		for (int iw = 0; iw < 512; ++iw) {
			for (int ic = 0; ic < 7; ++ic)
			{
				x[ic] = get_pixel(in_s, iw, ih, ic);  //whc
			}
			index = seven_ind_max(x);
			real_out.data[0 * 512 * 512 + ih * 512 + iw] = map[index][0];
			real_out.data[1 * 512 * 512 + ih * 512 + iw] = map[index][1];
			real_out.data[2 * 512 * 512 + ih * 512 + iw] = map[index][2];
		}
	}
	free_image(in_s);
	flip_image(real_out);
	return real_out;  //chw
}