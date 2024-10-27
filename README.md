# 学习tensorRT

## 1.准备环境
- [x] Ubuntu WSL2：22.04
- [x] CUDA：12.6
- [x] cuDNN：8.9.7.29
- [x] TensorRT：10.5
- [x] 最新的Nvidia显卡驱动

### 1.1 安装cuda和cudnn

安装前先安装cuda和对应版本的cuDnn，~~见：[learn-CUDA](https://github.com/Sknp1006/learn-CUDA#11%E5%AE%89%E8%A3%85cuda%E4%B8%8Ecudnn)~~ 现切换环境为windows的linux子系统

#### 1.1.1 安装cuda

> deb安装包：[CUDA Toolkit 12.6 Update 2 Downloads](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) 
>    - [通过百度网盘分享的文件：learn-tensorRT](https://pan.baidu.com/s/1u4g5aVwgQ2PgLhLu37dG4g?pwd=my1e) 
>
> 安装路径：/usr/local/cuda-12.6

- Installation Instructions：

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/  # *部分以实际为准
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

- 在 `.bashrc` 新增以添加cuda环境变量：

```bash
#begin env for cuda12.6
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.6
#end env cuda12.6
```

- 查看是否生效：

```bash
> source ./.bashrc
> nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Sep_12_02:18:05_PDT_2024
Cuda compilation tools, release 12.6, V12.6.77
Build cuda_12.6.r12.6/compiler.34841621_0
```

#### 1.1.2 安装cudnn

> deb安装包(cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb)：
>   - [cuDNN Download | NVIDIA Developer](https://developer.nvidia.com/rdp/cudnn-download)
>   - [通过百度网盘分享的文件：learn-tensorRT](https://pan.baidu.com/s/1u4g5aVwgQ2PgLhLu37dG4g?pwd=my1e) 

>
> 参考链接：[Installation Guide - NVIDIA Docs](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) 
>
> 安装路径：/usr
>
> 链接库目录：/usr/lib/x86_64-linux-gnu/
>
> sample安装路径：/usr/src/cudnn_samples_v8

```bash
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.7.29/cudnn-local-08A7D361-keyring.gpg /usr/share/keyrings/
sudo apt update

# 通过 search 找到这些包
# root@ADMIN-20230607I:~# apt search libcudnn8
# Sorting... Done
# Full Text Search... Done
# libcudnn8/unknown 8.9.7.29-1+cuda12.2 amd64
#   cuDNN runtime libraries

# libcudnn8-dev/unknown 8.9.7.29-1+cuda12.2 amd64
#   cuDNN development libraries and headers

# libcudnn8-samples/unknown 8.9.7.29-1+cuda12.2 amd64
#   cuDNN samples

sudo apt install libcudnn8=8.9.7.29-1+cuda12.2
sudo apt install libcudnn8-dev=8.9.7.29-1+cuda12.2
sudo apt install libcudnn8-samples=8.9.7.29-1+cuda12.2
```

- 查看libcudnn8的安装情况：

```bash
> dpkg-query -W libcudnn8
libcudnn8       8.9.7.29-1+cuda12.2
> dpkg-query -W libcudnn8-dev
libcudnn8-dev   8.9.7.29-1+cuda12.2
> dpkg-query -W libcudnn8-samples
libcudnn8-samples       8.9.7.29-1+cuda12.2
```

- 运行sample：

```bash
cd ~
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd $HOME/cudnn_samples_v8/mnistCUDNN
sudo apt install libfreeimage3 libfreeimage-dev
make clean && make
./mnistCUDNN

# 提示：Test passed! 则说明安装成功
```

### 1.2 下载TensorRT

<!-- >  deb安装包(nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.deb)：
>   - [TensorRT 8.6 GA for Ubuntu 22.04 and CUDA 12.0 and 12.1 DEB local repo Package](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.deb) -->
> deb安装包(nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.deb)：
> [TensorRT 10.5 GA for Ubuntu 22.04 and CUDA 12.6 DEB local repo Package](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.5.0/local_repo/nv-tensorrt-local-repo-ubuntu2204-10.5.0-cuda-12.6_1.0-1_amd64.deb)
>   - [通过百度网盘分享的文件：learn-tensorRT](https://pan.baidu.com/s/1u4g5aVwgQ2PgLhLu37dG4g?pwd=my1e) 
>
> 参考链接：[Debian Installation](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/install-guide/index.html#installing-debian) 
>
> 头文件路径：/usr/include/x86_64-linux-gnu
>
> 库文件路径：/usr/lib/x86_64-linux-gnu
>
> trtexec可执行文件路径：/usr/src/tensorrt/bin

#### 1.2.1 在Ubuntu安装

<!-- - 安装 `nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.deb` ：

```bash
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-12.0/*-keyring.gpg /usr/share/keyrings/ # *部分以实际为准
sudo apt update
``` -->

- 安装 `nv-tensorrt-local-repo-ubuntu2204-10.5.0-cuda-12.6_1.0-1_amd64.deb` ：

```bash
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-10.5.0-cuda-12.6_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.5.0-cuda-12.6/*-keyring.gpg /usr/share/keyrings/ # *部分以实际为准
sudo apt update
```

- 安装完整tensorrt：

```bash
sudo apt install tensorrt
```

- 验证安装：

```bash
> dpkg-query -W tensorrt
tensorrt        10.5.0.18-1+cuda12.6
```

- 将 `trtexec` 添加到 `.bashrc` 环境变量：

```bash
#begin env for trtexec
export PATH=/usr/src/tensorrt/bin:$PATH
#end env for trtexec

> source .bashrc
```

- 运行 `trtexec` 以验证是否正确安装


## 2.参考资料
- 官方：
  - [TensorRT 10.5.0 文档](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/index.html) 
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
