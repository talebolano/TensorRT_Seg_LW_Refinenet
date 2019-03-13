#ifdef OPENCV
//
// a single-threaded, multi client(using select), debug webserver - streaming out mjpg.
//  on win, _WIN32 has to be defined, must link against ws2_32.lib (socks on linux are for free)
//

//
// socket related abstractions:
//
#ifdef _WIN32
#pragma comment(lib, "ws2_32.lib")
#include <winsock.h>
#include <windows.h>
#include <time.h>
#define PORT        unsigned long
#define ADDRPOINTER   int*
struct _INIT_W32DATA
{
	WSADATA w;
	_INIT_W32DATA() { WSAStartup(MAKEWORD(2, 1), &w); }
} _init_once;
#else       /* ! win32 */
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#define PORT        unsigned short
#define SOCKET    int
#define HOSTENT  struct hostent
#define SOCKADDR    struct sockaddr
#define SOCKADDR_IN  struct sockaddr_in
#define ADDRPOINTER  unsigned int*
#define INVALID_SOCKET -1
#define SOCKET_ERROR   -1
#endif /* _WIN32 */

#include <cstdio>
#include <vector>
#include <iostream>
using std::cerr;
using std::endl;

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio.hpp"
#endif
using namespace cv;

#include "http_stream.h"
#include "image.h"

CvCapture* get_capture_video_stream(char *path) {
	CvCapture* cap = NULL;
	try {
		cap = (CvCapture*)new cv::VideoCapture(path);
	}
	catch (...) {
		std::cout << " Error: video-stream " << path << " can't be opened! \n";
	}
	return cap;
}

IplImage* get_webcam_frame(CvCapture *cap) {
	IplImage* src = NULL;
	try {
		cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
		cv::Mat frame;
		if (cpp_cap.isOpened())
		{
			cpp_cap >> frame;
			IplImage tmp = frame;
			src = cvCloneImage(&tmp);
			//src = &tmp;
		}
		else {
			std::cout << " Video-stream stoped! \n";
		}
	}
	catch (...) {
		std::cout << " Video-stream stoped! \n";
	}
	return src;
}


CvCapture* get_capture_webcam(int index) {
	CvCapture* cap = NULL;
	try {
		cap = (CvCapture*)new cv::VideoCapture(index);
		//((cv::VideoCapture*)cap)->set(CV_CAP_PROP_FRAME_WIDTH, 1280);
		//((cv::VideoCapture*)cap)->set(CV_CAP_PROP_FRAME_HEIGHT, 960);
	}
	catch (...) {
		std::cout << " Error: Web-camera " << index << " can't be opened! \n";
	}
	return cap;
}
extern "C" {
	image ipl_to_image(IplImage* src);    // image.c
}


image get_image_from_stream_cpp(CvCapture *cap)
{
	//IplImage* src = cvQueryFrame(cap);
	IplImage* src;//这一句造成了内存泄漏
	static int once = 1;
	if (once) {
		once = 0;
		do {
			src = get_webcam_frame(cap);//这一句造成了内存泄漏
			if (!src) return make_empty_image(0, 0, 0);
		} while (src->width < 1 || src->height < 1 || src->nChannels < 1);
		printf("Video stream: %d x %d \n", src->width, src->height);
	}
	else
		src = get_webcam_frame(cap);

	if (!src) {
		return make_empty_image(0, 0, 0);
	}
	image im = ipl_to_image(src);//重要
	rgbgr_image(im);//重要    原始bgr --》rgb
	cvReleaseImage(&src);
	return im;
}

#endif    // OPENCV