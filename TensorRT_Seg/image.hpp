#pragma once


typedef struct {
	int w;
	int h;
	int c;
	float *data;
} image;

float* normal(cv::Mat img);


cv::Mat read2mat(float * prob, cv::Mat out);


cv::Mat map2threeunchar(cv::Mat real_out, cv::Mat real_out_);
