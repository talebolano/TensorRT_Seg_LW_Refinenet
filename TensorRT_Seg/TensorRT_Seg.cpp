#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "common.h"
#include "image.h"
#include "NvInferPlugin.h"



#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio_c.h"
#endif

#include "http_stream.h"
#include "gettimeofday.h"
#include "BatchStream.h"

#endif
using namespace nvinfer1;

static const int INPUT_H = 512;
static const int INPUT_W = 512;
static const int INPUT_C = 3;
static const int OUTPUT_SIZE = 114688;
static Logger gLogger;
static int gUseDLACore{ -1 };
const char* INPUT_BLOB_NAME = "0";
//const char* OUTPUT_BLOB_NAME = "prob";



std::string locateFile(const std::string& input)
{

	std::vector<std::string> dirs;
	dirs.push_back(std::string("data/int8/") + std::string("/"));
	dirs.push_back(std::string("data/") + std::string("/"));
	return locateFile(input, dirs);

}

class Int8EntropyCalibrator : public IInt8EntropyCalibrator
{
public:
	//（1，3，512，512）--》（500，3，512，512）
	Int8EntropyCalibrator(batchstream& stream, bool readCache = true)
		: mStream(stream)
		, mReadCache(readCache)
	{
		//DimsNCHW dims = mStream.getDims();
		mInputCount = 1*3*512*512;
		CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
		//mStream.reset(firstBatch);
	}

	virtual ~Int8EntropyCalibrator()
	{
		CHECK(cudaFree(mDeviceInput));
	}

	int getBatchSize() const override { return mStream.getBatchSize(); }

	bool getBatch(void* bindings[], const char* names[], int nbBindings) override
	{
		if (!mStream.next())
			return false;

		CHECK(cudaMemcpy(mDeviceInput, mStream.get_image(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
		assert(!strcmp(names[0], INPUT_BLOB_NAME));
		bindings[0] = mDeviceInput;
		return true;
	}

	const void* readCalibrationCache(size_t& length) override     //读缓存
	{
		mCalibrationCache.clear();
		std::ifstream input(calibrationTableName(), std::ios::binary);
		input >> std::noskipws;
		if (mReadCache && input.good())
			std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

		length = mCalibrationCache.size();
		return length ? &mCalibrationCache[0] : nullptr;
	}

	void writeCalibrationCache(const void* cache, size_t length) override
	{
		std::ofstream output(calibrationTableName(), std::ios::binary);
		output.write(reinterpret_cast<const char*>(cache), length);
	}

private:
	static std::string calibrationTableName()
	{
		return std::string("CalibrationTable");
	}
	batchstream mStream;
	bool mReadCache{ true };

	size_t mInputCount;
	void* mDeviceInput{ nullptr };
	std::vector<char> mCalibrationCache;
};


void onnxToTRTModel(const std::string& modelFile, // name of the onnx model
	unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
	IHostMemory*& trtModelStream,
	DataType dataType,
	IInt8Calibrator* calibrator) // output buffer for the TensorRT model
{
	int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);
	nvinfer1::INetworkDefinition* network = builder->createNetwork();

	auto parser = nvonnxparser::createParser(*network, gLogger);


	//Optional - uncomment below lines to view network layer information
	//config->setPrintLayerInfo(true);
	//parser->reportParsingInfo();

	if (!parser->parseFromFile(modelFile.c_str(), verbosity))
	{
		string msg("failed to parse onnx file");
		gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
		exit(EXIT_FAILURE);
	}
	if ((dataType == DataType::kINT8 && !builder->platformHasFastInt8()) )
		exit(EXIT_FAILURE);  //如果不支持kint8或不支持khalf就返回false
	// Build the engine

	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(4_GB); //不能超过你的实际能用的显存的大小，例如我的1060的可用为4.98GB，超过4.98GB会报错
	builder->setInt8Mode(dataType == DataType::kINT8);  //
	builder->setInt8Calibrator(calibrator);  //
	samplesCommon::enableDLA(builder, gUseDLACore);
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we can destroy the parser
	parser->destroy();

	// serialize the engine, then close everything down  序列化
	trtModelStream = engine->serialize();
	engine->destroy();
	network->destroy();
	builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex, outputIndex;
	for (int b = 0; b < engine.getNbBindings(); ++b)
	{
		if (engine.bindingIsInput(b))
			inputIndex = b;
		else
			outputIndex = b;
	}
	// create GPU buffers and a stream   创建GPU缓冲区和流
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize *INPUT_C* INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize *INPUT_C* INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);//TensorRT的执行通常是异步的，因此将核加入队列放在CUDA流上
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{	
	batchstream calibrationStream(500);
	Int8EntropyCalibrator calibrator(calibrationStream);

	//gUseDLACore = samplesCommon::parseDLA(argc, argv);
	// create a TensorRT model from the onnx model and serialize it to a stream
	IHostMemory* trtModelStream{ nullptr };
	onnxToTRTModel("D:/pytorch/light-weight-refinenet/test_up.onnx", 1, trtModelStream, DataType::kINT8, &calibrator);  //读onnx模型,序列化引擎
	std::cout << "rialize model ready" << std::endl;
	assert(trtModelStream != nullptr);
	// deserialize the engine    DLA加速
	//反序列化引擎
	IRuntime* runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	if (gUseDLACore >= 0)
	{
		runtime->setDLACore(gUseDLACore);
	}
	//反序列化
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
	assert(engine != nullptr);
	trtModelStream->destroy();
	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);


	int cam_index = 0;
	char *filename = (argc > 1) ? argv[1] : 0;
	std::cout << "Hello World!\n";
	CvCapture * cap;

	if (filename) {
		//cap = cvCaptureFromFile(filename);
		cap = get_capture_video_stream(filename);
	}
	else
	{
		cap = get_capture_webcam(cam_index);;
	}
	cvNamedWindow("Segmentation", CV_WINDOW_NORMAL); //创建窗口显示图像，可以鼠标随意拖动窗口改变大小
	cvResizeWindow("Segmentation", 512, 512);//设定窗口大小
	float prob[OUTPUT_SIZE];
	float fps = 0;
	//for(int i =0;i<500;i++)
	while (1) 
	{
		struct timeval tval_before, tval_after, tval_result;
		gettimeofday(&tval_before, NULL);
		image in = get_image_from_stream_cpp(cap);//c,h,w结构且已经处以225，浮点数
		image in_s = resize_image(in, 512, 512);//改变图片大小为标准长宽，用于网络一的图片
		in_s = normal_image(in_s);//正则化
		//prob = 【1，7，128，128】-->imgae[7,128,128]-->【7,512，512】-->[3,512,512]
		//【512，512，7】-->[512,512,1]
		// run inference   进行推理
		doInference(*context, in_s.data, prob, 1);
		image real_out = Tranpose(prob); //[128，128，7]


		show_image(real_out, "Segmentation");   //显示图片

		free_image(in);
		free_image(in_s);
		free_image(real_out);
		if (cvWaitKey(10) == 27) break;
		gettimeofday(&tval_after, NULL);
		timersub(&tval_after, &tval_before, &tval_result);
		float curr = 1000000.f / ((long int)tval_result.tv_usec);
		printf("\nFPS:%.0f\n", fps);
		fps = .9*fps + .1*curr;
	}
	//cvdestroyAllWindows();
	cvDestroyAllWindows();
	shutdown_cap(cap);

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();
	std::cout << "shut down" << std::endl;
	//nvcaffeparser1::shutdownProtobufLibrary();

	return EXIT_SUCCESS;//无法退出？解决
}
