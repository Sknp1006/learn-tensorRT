# 学习tensorRT

## 1.准备环境
- [x] Ubuntu WSL2：22.04
- [x] CUDA：12.1
- [x] cuDNN：8.9.2.26
- [x] 最新的Nvidia显卡驱动

### 1.1 安装cuda和cudnn

安装前先安装cuda和对应版本的cuDnn，~~见：[learn-CUDA](https://github.com/Sknp1006/learn-CUDA#11%E5%AE%89%E8%A3%85cuda%E4%B8%8Ecudnn)~~ 现切换环境为windows的linux子系统

#### 1.1.1 安装cuda

> deb安装包：[CUDA Toolkit 12.1 Update 1 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) 
>
> 安装路径：/usr/local/cuda-12.1

- Installation Instructions：

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

- 在 `.bashrc` 新增以添加cuda环境变量：

```bash
#begin env for cuda12.1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.1
#end env cuda12.1
```

- 查看是否生效：

```bash
> source ./.bashrc
> nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```

#### 1.1.2 安装cudnn

> deb安装包：[cuDNN Download | NVIDIA Developer](https://developer.nvidia.com/rdp/cudnn-download) 
>
> 参考链接：[Installation Guide - NVIDIA Docs](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) 
>
> 安装路径：/usr
>
> 链接库目录：/usr/lib/x86_64-linux-gnu/
>
> sample安装路径：/usr/src/cudnn_samples_v8

```bash
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.2.26_1.0-1_amd64.deb

sudo apt update

sudo apt install libcudnn8=8.9.2.26-1+cuda12.1
sudo apt install libcudnn8-dev=8.9.2.26-1+cuda12.1
sudo apt install libcudnn8-samples=8.9.2.26-1+cuda12.1
```

- 查看libcudnn8的安装情况：

```bash
> apt search libcudnn
Sorting... Done
Full Text Search... Done
libcudnn8/unknown,now 8.9.2.26-1+cuda12.1 amd64 [installed]
  cuDNN runtime libraries

libcudnn8-dev/unknown,now 8.9.2.26-1+cuda12.1 amd64 [installed]
  cuDNN development libraries and headers

libcudnn8-samples/unknown,now 8.9.2.26-1+cuda12.1 amd64 [installed]
  cuDNN samples
```

- 运行sample：

```bash
cd ~
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd $HOME/cudnn_samples_v8/mnistCUDNN
sudo apt install libfreeimage3 libfreeimage-dev
make clean && make
./mnistCUDNN
```

### 1.2 安装TensorRT

>  deb安装包：[TensorRT 8.6 GA for Ubuntu 22.04 and CUDA 12.0 and 12.1 DEB local repo Package](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.deb) 
>
> 参考链接：[Debian Installation](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/install-guide/index.html#installing-debian) 
>
> 头文件路径：/usr/include/x86_64-linux-gnu
>
> 库文件路径：/usr/lib/x86_64-linux-gnu
>
> trtexec可执行文件路径：/usr/src/tensorrt/bin

#### 1.2.1 在Ubuntu安装

- 安装 `nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.deb` ：

```bash
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0/*-keyring.gpg /usr/share/keyrings/
sudo apt update
```

- 安装完整tensorrt：

```bash
sudo apt install tensorrt
```

- 验证安装：

```bash
> dpkg-query -W tensorrt
tensorrt        8.6.1.6-1+cuda12.0
```

- 将 `trtexec` 添加到 `.bashrc` 环境变量：

```bash
#begin env for trtexec
export PATH=/usr/src/tensorrt/bin${PATH:+:${PATH}}
#end env for trtexec
```

- 运行 `trtexec` 以验证是否正确安装

#### 1.2.2 在windows安装

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
