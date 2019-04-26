TensorRT for a simple segmentation model
=======================================
[![LICENSE](https://img.shields.io/badge/license-NPL%20(The%20996%20Prohibited%20License)-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)  <a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg" alt="996.icu"></a>

使用VOC Person Part 训练[LW Refinenet ——resnet50](https://github.com/DrSleep/light-weight-refinenet) 作为神经网络，生成onnx模型并导入c++中实现实时语义分割。

下载Resnet50 精度为64.1mIOU的onnx模型：[百度云](https://pan.baidu.com/s/18oCAH1Eu2fNwbtsek7av1w) 密码：kx0j 

## 效果
实验GPU：gtx1060  输入图像：512*512

###
|模型|速度|
|:-----:|--------|
|pytorch源码|11FPS|
|FP32|21FPS|
|INT8|32FPS|

###

## how to use

第一次推理，没有序列化模型：


TensorRT_Seg.exe no_have_serialize_txt  int8  save_serialize_name  here_your_video_file_name_or_cam here_your_onnxmodel_name here_your_Calibrator_file_name


TensorRT_Seg.exe no_have_serialize_txt  float32  save_serialize_name here_your_video_file_name_or_cam here_your_onnxmodel_name 


保存序列化模型后：


TensorRT_Seg.exe have_serialize_txt  int8  here_your_video_file_name_or_cam saved_serialize_name here_your_Calibrator_file_name


TensorRT_Seg.exe have_serialize_txt  float32  here_your_video_file_name_or_cam saved_serialize_name 


## Todo

~~1.实现int8推理~~

~~2.实现upsample插件层（目前使用反卷积层代替）~~(tensorRT5.1已有upsmaplenearest的操作，还是没有bilinear的操作)

3.实现现有模型的剪枝，并进行时间的比较


