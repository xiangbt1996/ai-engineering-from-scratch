# 数据管理

> 数据是燃料。你管理它的方式决定了你能跑多快。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 0，第 01 课
**时长：** 约 45 分钟

## 学习目标

- 使用 Hugging Face `datasets` 库加载、流式处理和缓存数据集
- 在 CSV、JSON、Parquet 和 Arrow 格式之间转换，并解释各自的优劣
- 使用固定随机种子创建可复现的训练集/验证集/测试集划分
- 使用 `.gitignore`、Git LFS 或 DVC 管理大型模型和数据集文件

## 为什么要做这件事

每个 AI 项目都从数据开始。你需要找到数据集、下载它们、在不同格式之间转换、为训练和评估进行划分，还要做版本管理以确保实验可复现。每次都手动操作既慢又容易出错。你需要一套可重复的工作流。

## 核心概念

```mermaid
graph TD
    A["Hugging Face Hub"] --> B["datasets 库"]
    B --> C["加载 / 流式处理"]
    C --> D["本地缓存<br/>~/.cache/huggingface/"]
    B --> E["格式转换<br/>CSV, JSON, Parquet, Arrow"]
    E --> F["数据划分<br/>train / val / test"]
    F --> G["你的训练流水线"]
```

Hugging Face `datasets` 库是加载 AI 数据的标准方式。它开箱即用地处理了下载、缓存、格式转换和流式加载。

## 动手搭建

### 第 1 步：安装 datasets 库

```bash
pip install datasets huggingface_hub
```

### 第 2 步：加载数据集

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset)
print(dataset["train"][0])
```

这会下载 IMDB 电影评论数据集。首次下载后，后续会从 `~/.cache/huggingface/datasets/` 的缓存中加载。

### 第 3 步：流式加载大型数据集

有些数据集太大，无法全部存在磁盘上。流式处理（Streaming）可以逐行加载，无需下载完整数据。

```python
dataset = load_dataset("wikimedia/wikipedia", "20220301.en", split="train", streaming=True)

for i, example in enumerate(dataset):
    print(example["title"])
    if i >= 4:
        break
```

流式模式返回一个 `IterableDataset`。你可以在数据到达时逐行处理。无论数据集多大，内存占用都保持恒定。

### 第 4 步：数据集格式

`datasets` 库底层使用 Apache Arrow。你可以根据流水线的需要转换为其他格式。

```python
dataset = load_dataset("imdb", split="train")

dataset.to_csv("imdb_train.csv")
dataset.to_json("imdb_train.json")
dataset.to_parquet("imdb_train.parquet")
```

格式对比：

| 格式 | 体积 | 读取速度 | 适用场景 |
|------|------|---------|---------|
| CSV | 大 | 慢 | 人类可读、电子表格 |
| JSON | 大 | 慢 | API、嵌套数据 |
| Parquet | 小 | 快 | 分析查询、列式操作 |
| Arrow | 小 | 最快 | 内存中处理（`datasets` 内部使用） |

在 AI 工作中，Parquet 是最佳存储格式。Arrow 是你在内存中使用的格式。CSV 和 JSON 用于数据交换。

### 第 5 步：数据划分

每个机器学习项目都需要三个划分：

- **训练集（Train）**：模型从中学习（通常占 80%）
- **验证集（Validation）**：训练过程中检查进度（通常占 10%）
- **测试集（Test）**：训练结束后的最终评估（通常占 10%）

有些数据集自带划分。如果没有，自己划分：

```python
dataset = load_dataset("imdb", split="train")

split = dataset.train_test_split(test_size=0.2, seed=42)
train_val = split["train"].train_test_split(test_size=0.125, seed=42)

train_ds = train_val["train"]
val_ds = train_val["test"]
test_ds = split["test"]

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
```

务必设置随机种子以确保可复现性。相同的种子每次都会产生相同的划分。

### 第 6 步：下载和缓存模型

模型文件很大。`huggingface_hub` 库负责下载和缓存管理。

```python
from huggingface_hub import hf_hub_download, snapshot_download

model_path = hf_hub_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    filename="config.json"
)
print(f"Cached at: {model_path}")

model_dir = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
print(f"Full model at: {model_dir}")
```

模型缓存在 `~/.cache/huggingface/hub/`。一旦下载完成，后续运行时会即时加载。

### 第 7 步：处理大文件

模型权重和大型数据集不应该放进 git。有三种选择：

**选项 A：.gitignore（最简单）**

```
*.bin
*.safetensors
*.pt
*.onnx
data/*.parquet
data/*.csv
models/
```

**选项 B：Git LFS（在 git 中追踪大文件）**

```bash
git lfs install
git lfs track "*.bin"
git lfs track "*.safetensors"
git add .gitattributes
```

Git LFS 在仓库中存储指针文件，实际文件存放在独立的服务器上。GitHub 免费提供 1 GB 额度。

**选项 C：DVC（数据版本控制）**

```bash
pip install dvc
dvc init
dvc add data/training_set.parquet
git add data/training_set.parquet.dvc data/.gitignore
git commit -m "Track training data with DVC"
```

DVC 创建小型 `.dvc` 文件指向你的数据。实际数据存放在 S3、GCS 或其他远程存储后端。

| 方案 | 复杂度 | 适用场景 |
|------|--------|---------|
| .gitignore | 低 | 个人项目、可重新获取的下载数据 |
| Git LFS | 中 | 团队通过 git 共享模型权重 |
| DVC | 高 | 可复现实验、大型数据集、团队协作 |

本课程中，`.gitignore` 就够用了。当你需要跨机器精确复现实验时，再使用 DVC。

### 第 8 步：存储模式

**本地存储**适用于 10 GB 以下的数据集。HF 缓存会自动处理。

**云存储**适用于更大的数据集或需要跨机器共享的场景：

```python
import os

local_path = os.path.expanduser("~/.cache/huggingface/datasets/")

# s3_path = "s3://my-bucket/datasets/"
# gcs_path = "gs://my-bucket/datasets/"
```

DVC 可以直接与 S3 和 GCS 集成：

```bash
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push
```

本课程中，本地存储就足够了。当你在远程 GPU 实例上进行微调时，云存储才变得重要。

## 本课程使用的数据集

| 数据集 | 所在课程 | 大小 | 教学内容 |
|--------|---------|------|---------|
| IMDB | 分词、分类 | 84 MB | 文本分类基础 |
| WikiText | 语言建模 | 181 MB | 下一个词预测 |
| SQuAD | 问答系统 | 35 MB | 问答、文本片段提取 |
| Common Crawl（子集） | 嵌入向量 | 不定 | 大规模文本处理 |
| MNIST | 视觉基础 | 21 MB | 图像分类基础 |
| COCO（子集） | 多模态 | 不定 | 图文配对 |

你现在不需要下载所有这些数据集。每节课会指定所需的数据。

## 实际使用

运行工具脚本验证一切正常：

```bash
python code/data_utils.py
```

这会下载一个小型数据集、进行格式转换、划分数据，并打印摘要信息。

## 交付物

本课产出：
- `code/data_utils.py` - 可复用的数据加载和缓存工具
- `outputs/prompt-data-helper.md` - 用于为任务找到合适数据集的提示词

## 练习

1. 使用 `mrpc` 配置加载 `glue` 数据集，并查看前 5 条样本
2. 流式加载 `c4` 数据集，计算 10 秒内能处理多少条样本
3. 将一个数据集转换为 Parquet 格式，与 CSV 比较文件大小
4. 使用固定种子创建 70/15/15 的训练/验证/测试划分，验证各划分的大小

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| 数据集划分（Dataset Split） | "训练数据" | 一个命名子集（train/val/test），在机器学习生命周期的不同阶段使用 |
| 流式处理（Streaming） | "懒加载" | 从远程源逐行处理数据，无需将完整数据集下载到本地 |
| Parquet | "压缩版 CSV" | 一种列式文件格式，针对分析查询和存储效率进行了优化 |
| Arrow | "快速数据帧" | 一种内存中的列式格式，datasets 库内部使用它实现零拷贝读取 |
| Git LFS | "大文件的 Git" | 一个扩展，将大文件存储在 git 仓库之外，同时在版本控制中保留指针 |
| DVC | "数据的 Git" | 一套面向数据集和模型的版本控制系统，与云存储集成 |
| 缓存（Cache） | "已经下载过了" | 之前获取的数据的本地副本，默认存储在 ~/.cache/huggingface/ |
