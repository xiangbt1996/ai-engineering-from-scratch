# 奇异值分解

> SVD 是线性代数的瑞士军刀。每个矩阵都有一个。每个数据科学家都需要一个。

**类型：** 动手构建
**语言：** Python, Julia
**前置课程：** 第一阶段，第 01 课（线性代数直觉）、第 02 课（向量与矩阵运算）、第 03 课（矩阵变换）
**时间：** 约 120 分钟

## 学习目标

- 通过幂迭代法实现 SVD，并解释 U、Sigma 和 V^T 的几何含义
- 应用截断 SVD（Truncated SVD）进行图像压缩，并衡量压缩比与重建误差的关系
- 通过 SVD 计算 Moore-Penrose 伪逆，用于求解超定最小二乘系统
- 将 SVD 与 PCA、推荐系统（潜在因子）以及 NLP 中的潜在语义分析联系起来

## 问题背景

你有一个 1000x2000 的矩阵。它可能是用户-电影评分矩阵，也可能是文档-词频表，还可能是一张图像的像素值。你需要对它进行压缩、去噪、发现其中的隐藏结构，或者用它来求解最小二乘系统。特征分解（Eigendecomposition）只适用于方阵，而且即便是方阵，也要求矩阵具有一组完整的线性无关特征向量。

SVD 适用于任意矩阵。任意形状。任意秩。无任何条件。它将矩阵分解为三个因子，揭示了矩阵对空间所做操作的几何本质。它是整个线性代数中最通用、最有用的分解方法。

## 概念讲解

### SVD 的几何意义

任何矩阵，无论形状如何，都按顺序执行三种操作：旋转、缩放、旋转。SVD 将这个分解过程显式化。

```
A = U * Sigma * V^T

      m x n     m x m    m x n    n x n
    （任意）   （旋转）  （缩放）  （旋转）
```

给定任意矩阵 A，SVD 将其分解为：
- V^T 在输入空间（n 维）中旋转向量
- Sigma 沿每个轴进行缩放（拉伸或压缩）
- U 将结果旋转到输出空间（m 维）

```mermaid
graph LR
    A["输入空间（n 维）\n数据云\n（任意方向）"] -->|"V^T\n（旋转）"| B["缩放空间\n与轴对齐\n然后被 Sigma 缩放"]
    B -->|"U\n（旋转）"| C["输出空间（m 维）\n旋转到输出\n方向"]
```

可以这样理解：你把一个矩阵交给 SVD，它告诉你："这个矩阵把一个输入球体，先用 V^T 旋转，再用 Sigma 拉伸成一个椭球体，最后用 U 旋转这个椭球体。" 奇异值就是椭球体各轴的长度。

### 完整分解

对于形状为 m x n 的矩阵 A：

```
A = U * Sigma * V^T

其中：
  U     是 m x m，正交矩阵（U^T U = I）
  Sigma 是 m x n，对角矩阵（对角线上是奇异值）
  V     是 n x n，正交矩阵（V^T V = I）

奇异值 sigma_1 >= sigma_2 >= ... >= sigma_r > 0
其中 r = rank(A)
```

U 的列称为左奇异向量。V 的列称为右奇异向量。Sigma 的对角元素称为奇异值（Singular Values）。奇异值始终非负，按照惯例按降序排列。

### 左奇异向量、奇异值、右奇异向量

SVD 的每个组成部分都有独特的几何含义。

**右奇异向量（V 的列）：** 它们构成输入空间（R^n）的一组标准正交基。它们是输入空间中被矩阵映射到输出空间正交方向的那些方向。可以把它们理解为定义域的自然坐标系。

**奇异值（Sigma 的对角线）：** 这些是缩放因子。第 i 个奇异值告诉你矩阵沿第 i 个右奇异向量方向拉伸了多少。奇异值为零意味着矩阵完全压碎了该方向。

**左奇异向量（U 的列）：** 它们构成输出空间（R^m）的一组标准正交基。第 i 个左奇异向量是第 i 个右奇异向量（经过 sigma_i 缩放后）在输出空间中的落点方向。

它们之间的关系：

```
A * v_i = sigma_i * u_i

矩阵 A 将第 i 个右奇异向量 v_i，
以 sigma_i 进行缩放，映射到第 i 个左奇异向量 u_i。
```

这为你提供了任何矩阵逐坐标操作的完整图景。

### 外积形式

SVD 可以写成秩为 1 的矩阵之和：

```
A = sigma_1 * u_1 * v_1^T + sigma_2 * u_2 * v_2^T + ... + sigma_r * u_r * v_r^T

每一项 sigma_i * u_i * v_i^T 是一个秩为 1 的矩阵（一个外积）。
完整矩阵是 r 个这样的矩阵之和，其中 r 是矩阵的秩。
```

这种形式是低秩近似的基础。每一项增加一层结构。第一项捕获最重要的单一模式。第二项捕获次重要的模式。以此类推。截断这个求和就能在任意给定秩下获得最优近似。

```
秩 1 近似：    A_1 = sigma_1 * u_1 * v_1^T
              （捕获主导模式）

秩 2 近似：    A_2 = sigma_1 * u_1 * v_1^T + sigma_2 * u_2 * v_2^T
              （捕获两个最重要的模式）

秩 k 近似：    A_k = 前 k 项之和
              （由 Eckart-Young 定理保证最优）
```

### 与特征分解的关系

SVD 和特征分解有着深刻的联系。A 的奇异值和奇异向量直接来自 A^T A 和 A A^T 的特征值和特征向量。

```
A^T A = V * Sigma^T * U^T * U * Sigma * V^T
      = V * Sigma^T * Sigma * V^T
      = V * D * V^T

其中 D = Sigma^T * Sigma 是对角矩阵，对角线上是 sigma_i^2。

所以：
- 右奇异向量（V）是 A^T A 的特征向量
- 奇异值的平方（sigma_i^2）是 A^T A 的特征值

类似地：
A A^T = U * Sigma * V^T * V * Sigma^T * U^T
      = U * Sigma * Sigma^T * U^T

所以：
- 左奇异向量（U）是 A A^T 的特征向量
- A A^T 的特征值也是 sigma_i^2
```

这个联系告诉你三件事：
1. 奇异值始终是实数且非负（它们是半正定矩阵特征值的平方根）。
2. 你可以通过 A^T A 的特征分解来计算 SVD，但这会将条件数平方，从而损失数值精度。专用的 SVD 算法会避免这个问题。
3. 当 A 是方阵且半正定时，SVD 和特征分解是同一件事。

### 截断 SVD：低秩近似

Eckart-Young-Mirsky 定理指出，对 A 的最优秩 k 近似（在 Frobenius 范数和谱范数下）可以通过仅保留前 k 个最大奇异值及其对应的向量来获得：

```
A_k = U_k * Sigma_k * V_k^T

其中：
  U_k     是 m x k  （U 的前 k 列）
  Sigma_k 是 k x k  （Sigma 的左上角 k x k 块）
  V_k     是 n x k  （V 的前 k 列）

近似误差 = sigma_{k+1}（谱范数下）
        = sqrt(sigma_{k+1}^2 + ... + sigma_r^2)（Frobenius 范数下）
```

这不仅仅是一个"不错的"近似。它是可以证明的秩 k 最优近似。没有其他秩 k 矩阵比它更接近 A。

| 分量 | 相对大小 | 在秩 3 近似中保留？ |
|-----------|-------------------|------------------------|
| sigma_1 | 最大 | 是 |
| sigma_2 | 大 | 是 |
| sigma_3 | 中等偏大 | 是 |
| sigma_4 | 中等 | 否（误差） |
| sigma_5 | 中等偏小 | 否（误差） |
| sigma_6 | 小 | 否（误差） |
| sigma_7 | 很小 | 否（误差） |
| sigma_8 | 极小 | 否（误差） |

保留前 3 个：A_3 捕获了三个最大的奇异值。误差 = 剩余的值（sigma_4 到 sigma_8）。

如果奇异值衰减很快，一个较小的 k 就能捕获矩阵的大部分信息。如果衰减缓慢，则该矩阵不具有低秩结构。

### 使用 SVD 进行图像压缩

灰度图像是一个像素强度矩阵。一张 800x600 的图像有 480,000 个值。SVD 允许你用少得多的值来近似它。

```
原始图像：800 x 600 = 480,000 个值

使用秩 k 的 SVD：
  U_k:      800 x k 个值
  Sigma_k:  k 个值
  V_k:      600 x k 个值
  总计:     k * (800 + 600 + 1) = k * 1401 个值

  k=10:   14,010 个值   （原始的 2.9%）
  k=50:   70,050 个值   （原始的 14.6%）
  k=100: 140,100 个值   （原始的 29.2%）

  k 越小，压缩比越好，
  但视觉质量会下降。
```

关键洞察：自然图像的奇异值衰减非常快。前几个奇异值捕获了宏观结构（形状、渐变）。后面的奇异值捕获的是细节和噪声。在秩 50 处截断通常就能产生与原图几乎一模一样的图像，同时节省 85% 的存储空间。

### SVD 用于推荐系统

Netflix Prize 使这一方法名声大噪。你有一个用户-电影评分矩阵，其中大部分条目是缺失的。

```
             电影1   电影2   电影3   电影4   电影5
  用户1      [  5      ?       3       ?       1  ]
  用户2      [  ?      4       ?       2       ?  ]
  用户3      [  3      ?       5       ?       ?  ]
  用户4      [  ?      ?       ?       4       3  ]

  ? = 未知评分
```

核心思路：这个评分矩阵具有低秩特性。用户的偏好并非完全独立。只有少数几个潜在因子（Latent Factors）（如动作片 vs. 剧情片、新片 vs. 老片、烧脑 vs. 直觉型）就能解释大部分偏好。

对（填充后的）评分矩阵进行 SVD 分解：
- U：潜在因子空间中的用户画像
- Sigma：每个潜在因子的重要性
- V^T：潜在因子空间中的电影画像

用户对某部电影的预测评分就是其用户画像与电影画像的点积（加上奇异值的权重）。低秩近似填补了缺失的条目。

在实际应用中，人们使用变体方法，如 Simon Funk 的增量 SVD 或交替最小二乘法（ALS），它们可以直接处理缺失数据。但核心思想是相同的：通过 SVD 进行潜在因子分解。

### SVD 在 NLP 中的应用：潜在语义分析

潜在语义分析（Latent Semantic Analysis, LSA），也称为潜在语义索引（Latent Semantic Indexing, LSI），将 SVD 应用于词-文档矩阵。

```
             文档1   文档2   文档3   文档4
  "cat"      [  3      0      1      0  ]
  "dog"      [  2      0      0      1  ]
  "fish"     [  0      4      1      0  ]
  "pet"      [  1      1      1      1  ]
  "ocean"    [  0      3      0      0  ]

经过秩 k=2 的 SVD 后：

  每个文档变成二维"概念空间"中的一个点。
  每个词也变成同一二维空间中的一个点。
  主题相似的文档会聚集在一起。
  含义相近的词会聚集在一起。

  "cat" 和 "dog" 最终靠近彼此（陆地宠物）。
  "fish" 和 "ocean" 最终靠近彼此（水的概念）。
  文档1 和 文档3 如果主题相似就会聚类在一起。
```

LSA 是最早成功从原始文本中捕获语义相似性的方法之一。它之所以有效，是因为同义词往往出现在相似的文档中，所以 SVD 将它们归入了相同的潜在维度。现代词嵌入（Word2Vec, GloVe）可以看作是这一思想的后继者。

### SVD 用于降噪

带噪数据的信号集中在前几个最大的奇异值中，而噪声分散在所有奇异值上。截断操作可以去除噪声底面。

**纯净信号的奇异值：**

| 分量 | 大小 | 类型 |
|-----------|-----------|------|
| sigma_1 | 很大 | 信号 |
| sigma_2 | 大 | 信号 |
| sigma_3 | 中等 | 信号 |
| sigma_4 | 接近零 | 可忽略 |
| sigma_5 | 接近零 | 可忽略 |

**含噪信号的奇异值（噪声叠加到所有分量上）：**

| 分量 | 大小 | 类型 |
|-----------|-----------|------|
| sigma_1 | 很大 | 信号 |
| sigma_2 | 大 | 信号 |
| sigma_3 | 中等 | 信号 |
| sigma_4 | 小 | 噪声 |
| sigma_5 | 小 | 噪声 |
| sigma_6 | 小 | 噪声 |
| sigma_7 | 小 | 噪声 |

```mermaid
graph TD
    A["所有奇异值"] --> B{"有明显间隔？"}
    B -->|"间隔以上"| C["信号：保留这些（前 k 个）"]
    B -->|"间隔以下"| D["噪声：丢弃这些"]
    C --> E["用 A_k 重建，得到去噪版本"]
```

这在信号处理、科学测量和数据清洗中广泛使用。任何时候你有一个被加性噪声污染的矩阵，截断 SVD 都是分离信号和噪声的一种有原则的方法。

### 通过 SVD 计算伪逆

Moore-Penrose 伪逆 A+ 将矩阵求逆推广到非方阵和奇异矩阵。SVD 使其计算变得简单。

```
如果 A = U * Sigma * V^T，则：

A+ = V * Sigma+ * U^T

其中 Sigma+ 的构造方法为：
  1. 转置 Sigma（交换行和列）
  2. 将每个非零对角元素 sigma_i 替换为 1/sigma_i
  3. 零保持为零

对于 A（m x n）：      A+ 是（n x m）
对于 Sigma（m x n）：  Sigma+ 是（n x m）
```

伪逆用于求解最小二乘问题。如果 Ax = b 没有精确解（超定系统），那么 x = A+ b 就是最小二乘解（最小化 ||Ax - b||）。

```
超定系统（方程数多于未知数）：

  [1  1]         [3]
  [2  1] x   =   [5]       不存在精确解。
  [3  1]         [6]

  x_ls = A+ b = V * Sigma+ * U^T * b

  这给出使残差平方和最小的 x。
  与正规方程 (A^T A)^(-1) A^T b 的结果相同，
  但数值上更稳定。
```

### 数值稳定性优势

计算 A^T A 的特征分解会将奇异值平方（A^T A 的特征值是 sigma_i^2）。这会将条件数（Condition Number）平方，放大数值误差。

```
示例：
  A 的奇异值为 [1000, 1, 0.001]
  A 的条件数：1000 / 0.001 = 10^6

  A^T A 的特征值为 [10^6, 1, 10^{-6}]
  A^T A 的条件数：10^6 / 10^{-6} = 10^{12}

  直接计算 SVD：处理条件数 10^6
  通过 A^T A 计算：处理条件数 10^{12}
                    （额外损失 6 位精度）
```

现代 SVD 算法（Golub-Kahan 双对角化）直接在 A 上工作，从不构造 A^T A。这就是为什么你应该始终优先使用 `np.linalg.svd(A)` 而不是 `np.linalg.eig(A.T @ A)`。

### 与 PCA 的联系

PCA 就是对中心化数据做 SVD。这不是类比，而是字面意义上完全相同的计算。

```
给定数据矩阵 X（n_samples x n_features），已中心化（减去均值）：

协方差矩阵：C = (1/(n-1)) * X^T X

PCA 寻找 C 的特征向量。但是：

  X = U * Sigma * V^T    （X 的 SVD）

  X^T X = V * Sigma^2 * V^T

  C = (1/(n-1)) * V * Sigma^2 * V^T

所以主成分恰好就是右奇异向量 V。
每个主成分的解释方差为 sigma_i^2 / (n-1)。

在 sklearn 中，PCA 使用 SVD 实现，而非特征分解。
这更快且数值上更稳定。
```

这意味着你在第 10 课中学到的关于降维的一切，底层都是 SVD。PCA 是 SVD 在机器学习中最常见的应用。

## 动手构建

### 第 1 步：使用幂迭代法从零实现 SVD

思路：要找到最大的奇异值及其向量，对 A^T A（或 A A^T）使用幂迭代法。然后对矩阵进行收缩，重复该过程找下一个奇异值。

```python
import numpy as np

def power_iteration(M, num_iters=100):
    n = M.shape[1]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iters):
        Mv = M @ v
        v = Mv / np.linalg.norm(Mv)

    eigenvalue = v @ M @ v
    return eigenvalue, v

def svd_from_scratch(A, k=None):
    m, n = A.shape
    if k is None:
        k = min(m, n)

    sigmas = []
    us = []
    vs = []

    A_residual = A.copy().astype(float)

    for _ in range(k):
        AtA = A_residual.T @ A_residual
        eigenvalue, v = power_iteration(AtA, num_iters=200)

        if eigenvalue < 1e-10:
            break

        sigma = np.sqrt(eigenvalue)
        u = A_residual @ v / sigma

        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        A_residual = A_residual - sigma * np.outer(u, v)

    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)
    V = np.column_stack(vs) if vs else np.empty((n, 0))

    return U, S, V
```

### 第 2 步：测试并与 NumPy 比较

```python
np.random.seed(42)
A = np.random.randn(5, 4)

U_ours, S_ours, V_ours = svd_from_scratch(A)
U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

print("Our singular values:", np.round(S_ours, 4))
print("NumPy singular values:", np.round(S_np, 4))

A_reconstructed = U_ours @ np.diag(S_ours) @ V_ours.T
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.8f}")
```

### 第 3 步：图像压缩演示

```python
def compress_image_svd(image_matrix, k):
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return compressed

image = np.random.seed(42)
rows, cols = 200, 300
image = np.random.randn(rows, cols)

for k in [1, 5, 10, 20, 50]:
    compressed = compress_image_svd(image, k)
    error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
    original_size = rows * cols
    compressed_size = k * (rows + cols + 1)
    ratio = compressed_size / original_size
    print(f"k={k:>3d}  error={error:.4f}  storage={ratio:.1%}")
```

### 第 4 步：降噪

```python
np.random.seed(42)
clean = np.outer(np.sin(np.linspace(0, 4*np.pi, 100)),
                 np.cos(np.linspace(0, 2*np.pi, 80)))
noise = 0.3 * np.random.randn(100, 80)
noisy = clean + noise

U, S, Vt = np.linalg.svd(noisy, full_matrices=False)
denoised = U[:, :5] @ np.diag(S[:5]) @ Vt[:5, :]

print(f"Noisy error:    {np.linalg.norm(noisy - clean):.4f}")
print(f"Denoised error: {np.linalg.norm(denoised - clean):.4f}")
print(f"Improvement:    {(1 - np.linalg.norm(denoised - clean) / np.linalg.norm(noisy - clean)):.1%}")
```

### 第 5 步：伪逆

```python
A = np.array([[1, 1], [2, 1], [3, 1]], dtype=float)
b = np.array([3, 5, 6], dtype=float)

U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.diag(1.0 / S)
A_pinv = Vt.T @ S_inv @ U.T

x_svd = A_pinv @ b
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
x_pinv = np.linalg.pinv(A) @ b

print(f"SVD pseudoinverse solution:  {x_svd}")
print(f"np.linalg.lstsq solution:   {x_lstsq}")
print(f"np.linalg.pinv solution:    {x_pinv}")
```

## 使用方式

完整的可运行演示在 `code/svd.py` 中。运行它可以看到 SVD 应用于图像压缩、推荐系统、潜在语义分析和降噪的效果。

```bash
python svd.py
```

Julia 版本在 `code/svd.jl` 中，使用 Julia 原生的 `svd()` 函数和 `LinearAlgebra` 包演示了相同的概念。

```bash
julia svd.jl
```

## 交付产出

本课程产出：
- `outputs/skill-svd.md` —— 一个帮助你了解何时以及如何在实际项目中应用 SVD 的技能文档

## 练习

1. 不使用幂迭代法，从零实现完整的 SVD。改用 A^T A 的特征分解来获取 V 和奇异值，然后计算 U = A V Sigma^{-1}。将数值精度与你的幂迭代版本和 NumPy 进行比较。

2. 加载一张真实的灰度图像（或将一张图像转换为灰度）。在秩 1、5、10、25、50、100 下进行压缩。对每个秩，计算压缩比和相对误差。找到图像在视觉上可接受的秩。

3. 构建一个小型推荐系统。创建一个 10x8 的用户-电影评分矩阵，其中有一些已知条目。用行均值填充缺失条目。计算 SVD 并重建秩 3 的近似。使用重建矩阵来预测缺失评分。验证预测是否合理。

4. 创建一个 100x50 的文档-词矩阵，包含 3 个合成主题。每个主题有 5 个相关词。添加噪声。应用 SVD 并验证前 3 个奇异值远大于其余的。将文档投影到 3D 潜在空间中，检查来自同一主题的文档是否聚类在一起。

5. 生成一个纯净的低秩矩阵（秩 3，大小 50x40），并在不同水平下添加高斯噪声（sigma = 0.1, 0.5, 1.0, 2.0）。对每个噪声水平，通过遍历 k 从 1 到 40 来寻找最优截断秩，衡量相对于纯净矩阵的重建误差。绘图展示最优 k 如何随噪声水平变化。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------------|----------------------|
| SVD | "分解任意矩阵" | 将 A 分解为 U Sigma V^T，其中 U 和 V 是正交矩阵，Sigma 是非负对角矩阵。适用于任意形状的任意矩阵。 |
| 奇异值 | "这个分量有多重要" | Sigma 的第 i 个对角元素。衡量矩阵沿第 i 个主方向拉伸的程度。始终非负，按降序排列。 |
| 左奇异向量 | "输出方向" | U 的一列。第 i 个右奇异向量（经过 sigma_i 缩放后）在输出空间中映射到的方向。 |
| 右奇异向量 | "输入方向" | V 的一列。输入空间中被矩阵映射到第 i 个左奇异向量（经过 sigma_i 缩放后）的方向。 |
| 截断 SVD | "低秩近似" | 仅保留前 k 个最大奇异值及其向量。产生可证明的秩 k 最优近似（Eckart-Young 定理）。 |
| 秩 | "真实维度" | 非零奇异值的个数。告诉你矩阵实际使用了多少个独立方向。 |
| 伪逆 | "广义逆" | V Sigma+ U^T。对非零奇异值取倒数，零保持为零。为非方阵或奇异矩阵求解最小二乘问题。 |
| 条件数 | "对误差的敏感程度" | sigma_max / sigma_min。条件数越大意味着输入的微小变化会引起输出的巨大变化。SVD 直接揭示了这一点。 |
| 潜在因子 | "隐藏变量" | SVD 发现的低秩空间中的一个维度。在推荐系统中，潜在因子可能对应类型偏好。在 NLP 中，可能对应主题。 |
| Frobenius 范数 | "矩阵总大小" | 所有元素平方和的平方根。等于所有奇异值平方和的平方根。用于衡量近似误差。 |
| Eckart-Young 定理 | "SVD 给出最优压缩" | 对于任意目标秩 k，截断 SVD 在所有可能的秩 k 矩阵中最小化近似误差。 |
| 幂迭代法 | "找到最大的特征向量" | 反复用矩阵乘以一个随机向量并归一化。收敛到最大特征值对应的特征向量。是许多 SVD 算法的构建模块。 |

## 延伸阅读

- [Gilbert Strang: Linear Algebra and Its Applications, Chapter 7](https://math.mit.edu/~gs/linearalgebra/) - SVD 的系统讲解及应用
- [3Blue1Brown: But what is the SVD?](https://www.youtube.com/watch?v=vSczTbgc8Rc) - SVD 的几何直觉
- [We Recommend a Singular Value Decomposition](https://www.ams.org/publicoutreach/feature-column/fcarc-svd) - 美国数学学会的通俗概述
- [Netflix Prize and Matrix Factorization](https://sifter.org/~simon/journal/20061211.html) - Simon Funk 关于 SVD 用于推荐系统的原始博客文章
- [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) - SVD 在 NLP 中的原始应用
- [Numerical Linear Algebra by Trefethen and Bau](https://people.maths.ox.ac.uk/trefethen/text.html) - 理解 SVD 算法及其数值性质的权威参考
