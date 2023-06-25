# 学习tensorRT

## 1.准备环境
- [x] 操作系统：Windows10
- [x] CUDA：11.6
- [x] cuDNN：8.9
- [x] 最新的Nvidia显卡驱动
  
### 1.1.安装TensorRT
> 安装前先安装cuda和对应版本的cuDnn，见：[learn-CUDA](https://github.com/Sknp1006/learn-CUDA#11%E5%AE%89%E8%A3%85cuda%E4%B8%8Ecudnn) 
- 下载对应版本的TensorRT：[百度网盘](https://pan.baidu.com/s/1sEN0m-OYk5cieKxQkVd8WQ?pwd=xqkh) 
- 将压缩包解压到某个目录，例如 `C盘根目录` 
- 创建用户环境变量：
  - 变量名：TensorRT_HOME
  - 变量值：C:\TensorRT-8.6.1.6
- 添加用户 `Path` 变量：
  - 新建：%TensorRT_HOME%\bin
- 运行 `trtexec` 以验证是否正确安装

## 2.参考资料
- 官方：
  - [trt-samples-for-hackathon-cn](https://github.com/NVIDIA/trt-samples-for-hackathon-cn) 
  - [Quick Start Guide :: NVIDIA Deep Learning TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/) 
  - [Speeding Up Deep Learning Inference Using NVIDIA TensorRT (Updated) | NVIDIA Technical Blog](https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorrt-updated/) 
  - 插件库：
    - [TensorRT/plugin at master · NVIDIA/TensorRT · GitHub](https://github.com/NVIDIA/TensorRT/tree/master/plugin) 
- 竞赛
  - [英伟达TensorRT加速AI推理 Hackathon 2022 —— Transformer模型优化赛](https://tianchi.aliyun.com/competition/entrance/531953/information) 
- 其他
  - [模型部署 | Linux环境配置安装TensorRT保姆级教程_linux安装tensorrt_自动驾驶之心的博客-CSDN博客](https://blog.csdn.net/CV_Autobot/article/details/129020014) 
  - [【7.0】实现TensorRT自定义插件(plugin)自由！](https://zhuanlan.zhihu.com/p/297002406) 


## 3.其他

chatGPT Says:
```txt
当你准备学习TensorRT和CUDA时，以下是一个合理的学习顺序建议：

CUDA基础知识：
在开始学习TensorRT之前，建议先掌握CUDA的基础知识，包括CUDA编程模型、CUDA线程和块、内存管理、CUDA核函数编写等。你可以阅读NVIDIA的CUDA文档和教程，进行实践和编程练习。

深度学习基础知识：
TensorRT主要用于深度学习推理，因此建议对深度学习的基本概念和常用框架（如TensorFlow或PyTorch）有一定的了解。这包括神经网络的基本原理、常见的深度学习模型和网络结构、模型训练和优化等。

TensorRT入门：
了解TensorRT的基本概念和功能，包括推理引擎、网络优化、精度校准、TensorRT API等。可以从NVIDIA官方文档和教程开始，了解TensorRT的安装、配置和基本用法。

模型转换和优化：
学习如何将训练好的深度学习模型转换为TensorRT可用的格式，包括常见的模型格式（如ONNX、TensorFlow SavedModel）的转换和优化过程。掌握TensorRT的网络优化技术，如层融合、量化、剪枝等，以提高推理性能和效率。

TensorRT高级特性：
学习TensorRT的高级特性和功能，如动态形状支持、多个推理引擎的管理、批量推理、流式推理等。了解如何在TensorRT中使用插件和自定义层，以适应特定的应用需求。

性能优化和调试：
学习如何优化和调试TensorRT推理过程，包括利用TensorRT Profiler进行性能分析、使用NVIDIA Nsight工具进行GPU性能分析、内存优化和调试技巧等。

这只是一个大致的学习顺序建议，具体的学习过程可能会因个人需求和学习进度而有所调整。在学习过程中，参考官方文档、教程和示例代码是非常重要的，同时可以参与社区讨论、阅读相关的论文和博客，以扩展对TensorRT和CUDA的理解和应用。
```