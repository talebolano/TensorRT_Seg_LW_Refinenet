#pragma once

#ifdef __cplusplus
extern "C" {
#endif

	//你自己写的函数声明的头文件也写了函数定义的cpp文件也加入工程了
	//而且你很确定函数体肯定是在这个库文件中，却依然出现LNK2019错误。
	//C语言和C++语言混编，因为C++支持函数重载所以C++编译器生成的库文件中的函数名会面目全非，
	//解决方法 ，在c 头文件加入如上
#ifndef IMAGE_H
#define IMAGE_H

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>

	typedef struct {
		int w;
		int h;
		int c;
		float *data;
	} image;
	void free_image(image m);
	void show_image(image p, const char *name);
	image make_empty_image(int w, int h, int c);//73
	image make_image(int w, int h, int c);  //71
	void rgbgr_image(image im);
	image copy_image(image p);
	static float get_pixel(image m, int x, int y, int c);
	void constrain_image(image im);
	image resize_image(image im, int w, int h);
	image normal_image(image im);
	image Tranpose(float *prob);
	void flip_image(image a);
	//image ipl_to_image(IplImage* src); //
	//void rgbgr_image(image im);
	//image get_image_from_stream_cpp(CvCapture *cap);

#endif

#ifdef __cplusplus
}
#endif#pragma once

