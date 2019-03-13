# TensorRT for a simple segmention model

使用VOC Person Part 训练[LW Refinenet ——resnet50](https://github.com/DrSleep/light-weight-refinenet) 作为神经网络，生成onnx模型并导入c++中实现实时语义分割。

下载onnx模型：[百度云](https://pan.baidu.com/s/18oCAH1Eu2fNwbtsek7av1w) 密码：kx0j

## Todo

1.实现int8推理

2.实现upsample插件层（目前使用反卷积层代替）
