# GPU 设置与云计算

> 用 CPU 学习没问题，但真正训练模型需要 GPU。

**类型：** 动手搭建  
**语言：** Python  
**前置要求：** Phase 0, Lesson 01  
**时间：** 约 45 分钟

## 学习目标

- 用 `nvidia-smi` 和 PyTorch 的 CUDA API 验证本地 GPU 是否可用
- 配置 Google Colab 的免费 T4 GPU 进行云端实验
- 对比 CPU 与 GPU 的矩阵乘法速度差异
- 用 fp16 经验法则估算你的显存能装多大的模型

## 为什么要做这件事

Phase 1-3 的大部分课程在 CPU 上就能跑。但一旦开始训练 CNN、Transformer 或 LLM（Phase 4 以后），你就需要 GPU 加速。一个在 CPU 上跑 8 小时的训练任务，在 GPU 上可能只需 10 分钟。

你有三种选择：本地 GPU、云 GPU 或 Google Colab（免费）。

## 核心概念

```
你的选项：

1. 本地 NVIDIA GPU
   费用：$0（如果你已经有）
   配置：安装 CUDA + cuDNN
   适合：日常使用、大数据集

2. Google Colab（免费额度）
   费用：$0
   配置：无需配置
   适合：快速实验、家里没有 GPU

3. 云 GPU（Lambda、RunPod、Vast.ai）
   费用：$0.20-2.00/小时
   配置：SSH + 安装环境
   适合：正式训练、大模型
```

## 动手搭建

### 选项 1：本地 NVIDIA GPU

先检查你是否有显卡：

```bash
nvidia-smi
```

安装带 CUDA 支持的 PyTorch：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 选项 2：Google Colab

1. 打开 [colab.research.google.com](https://colab.research.google.com)
2. 菜单：运行时 > 更改运行时类型 > T4 GPU
3. 运行 `!nvidia-smi` 验证

你可以把本课程的 notebook 直接上传到 Colab 运行。

### 选项 3：云 GPU

Lambda Labs、RunPod 或 Vast.ai：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有 GPU？没关系

大部分课程在 CPU 上就能完成。需要 GPU 的课程会标注说明并提供 Colab 链接。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 动手实验：GPU vs CPU 性能对比

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 练习

1. 运行上面的性能对比代码，比较 CPU 和 GPU 的耗时
2. 如果没有 GPU，在 Google Colab 上跑一遍再对比
3. 查看你有多少显存，用经验法则估算能装多大的模型（fp16 下每个参数占 2 字节）

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| CUDA | "GPU 编程" | NVIDIA 的并行计算平台，让你能在 GPU 上运行代码 |
| VRAM | "显存" | GPU 上的显存，和系统内存（RAM）独立，决定了能加载多大的模型 |
| fp16 | "半精度" | 16 位浮点数，内存占用是 fp32 的一半，精度损失很小 |
| Tensor Core | "矩阵加速硬件" | GPU 上专门做矩阵乘法的计算单元，比普通核心快 4-8 倍 |
