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
  - 