# 数值稳定性

> 浮点数是一个有漏洞的抽象。它会在训练过程中坑你一把，而且你完全看不到它的到来。

**类型：** 动手构建
**语言：** Python
**前置课程：** 第一阶段，第 01-04 课
**时间：** 约 120 分钟

## 学习目标

- 使用最大值减法技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点运算中的溢出（Overflow）、下溢（Underflow）和灾难性抵消（Catastrophic Cancellation）
- 使用中心差分法验证解析梯度与数值梯度的一致性
- 解释为什么 bfloat16 在训练中优于 float16，以及损失缩放如何防止梯度下溢

## 问题背景

你的模型训练了三个小时，然后 loss 变成了 NaN。你加了一行 print 语句，发现在第 9,000 步时 logits 还是正常的，到了第 9,001 步就变成了 `inf`，到第 9,002 步所有梯度都变成了 `nan`，训练彻底死掉了。

或者：你的模型训练完成了，但精度比论文声称的低了 2%。你检查了所有东西——架构一致、超参数一致、数据一致。问题在于论文用的是 float32，而你用了 float16 却没有正确的缩放。三十二位累积的舍入误差悄悄吃掉了你的精度。

又或者：你从零实现了交叉熵损失。在小 logits 上能正常工作，但当 logits 超过 100 时，它返回 `inf`。softmax 溢出了，因为 `exp(100)` 超出了 float32 能表示的范围。每个 ML 框架都用一个两行的技巧来处理这个问题，而你不知道这个技巧的存在。

数值稳定性不是一个理论问题，它是训练成功与静默失败之间的分水岭。你将来调试的每一个严重的 ML bug，最终都会归结到浮点数。

## 核心概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准将实数存储为浮点值。一个浮点数由三部分组成：符号位（Sign Bit）、指数（Exponent）和尾数（Mantissa，也叫有效数字 Significand）。

```
Float32 布局（共 32 位）：
[1 符号位] [8 指数位] [23 尾数位]

值 = (-1)^符号 * 2^(指数 - 127) * 1.尾数
```

尾数决定精度（有多少位有效数字），指数决定范围（能表示多大或多小的数）。

```
格式       位数   指数位  尾数位   十进制精度位数   范围（近似）
float64    64     11      52      ~15-16          +/- 1.8e308
float32    32     8       23      ~7-8            +/- 3.4e38
float16    16     5       10      ~3-4            +/- 65,504
bfloat16   16     8       7       ~2-3            +/- 3.4e38
```

float32 大约给你 7 位十进制精度。这意味着它能区分 1.0000001 和 1.0000002，但分不清 1.00000001 和 1.00000002。超过 7 位之后，一切都是舍入噪声。

float16 大约给你 3 位精度。它能表示的最大数是 65,504。对于 ML 来说这小得令人不安——logits、梯度和激活值经常超过这个数。

bfloat16 是 Google 对 float16 范围问题的回答。它和 float32 有相同的 8 位指数（相同的范围，可达 3.4e38），但只有 7 位尾数（精度比 float16 还低）。对于训练神经网络来说，范围比精度更重要，所以 bfloat16 通常胜出。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 无法在二进制浮点中精确表示。在二进制中，它是一个无限循环小数：

```
0.1 的二进制 = 0.0001100110011001100110011...（无限循环）
```

float32 把它截断到 23 位尾数。存储的值约为 0.100000001490116。类似地，0.2 约存储为 0.200000002980232。它们的和是 0.300000004470348，而不是 0.3。

```
在 Python 中：
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对 ML 的影响：

1. 像 `if loss < threshold` 这样的 loss 比较可能给出错误答案
2. 累积大量小值（数千步的梯度更新）会偏离真实求和
3. 校验和和可复现性测试在用 `==` 比较浮点数时会失败

解决方法：永远不要用 `==` 比较浮点数。使用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### 灾难性抵消

当你对两个几乎相等的浮点数做减法时，有效数字相互抵消，舍入噪声被提升到了前导数字的位置。

```
a = 1.0000001    （在 float32 中存储为 1.00000011920929）
b = 1.0000000    （在 float32 中存储为 1.00000000000000）

真实差值：  0.0000001
计算结果：  0.00000011920929

相对误差：19.2%
```

一次减法就产生了 19% 的相对误差。在 ML 中，以下情况会出现这个问题：

- 计算均值较大的数据的方差：当 E[x] 很大时用 `E[x^2] - E[x]^2`
- 对接近相等的对数概率做减法
- 用过小的 epsilon 计算有限差分梯度

解决方法：重新排列公式，避免对大的、近似相等的数做减法。对于方差，使用 Welford 算法或先对数据做中心化。对于对数概率，全程在对数空间中操作。

### 溢出和下溢

溢出发生在结果太大而无法表示时。下溢发生在结果太小时（比最小可表示正数更接近零）。

```
Float32 边界：
  最大值：  3.4028235e+38
  最小正数（正规数）：1.175e-38
  最小正数（非正规数）：1.401e-45
  溢出：  任何 > 3.4e38 的数变为 inf
  下溢：  任何 < 1.4e-45 的数变为 0.0
```

`exp()` 函数是 ML 中溢出的主要来源：

```
exp(88.7)  = 3.40e+38   （勉强在 float32 范围内）
exp(89.0)  = inf         （溢出）
exp(-87.3) = 1.18e-38   （刚好在下溢边界以上）
exp(-104)  = 0.0         （下溢为零）
```

`log()` 函数则在另一个方向出问题：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      （没问题）
log(1e-46) = -inf        （输入先下溢为 0，然后 log(0) = -inf）
```

在 ML 中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在交叉熵、对数似然和 KL 散度中。`log(exp(x))` 的组合如果没有正确的技巧，就是一个雷区。

### Log-Sum-Exp 技巧

直接计算 `log(sum(exp(x_i)))` 在数值上是危险的。如果任何 `x_i` 很大，`exp(x_i)` 就会溢出。如果所有 `x_i` 都非常负，每个 `exp(x_i)` 都下溢为零，`log(0)` 就是 `-inf`。

技巧：在取指数之前减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么这有效：减去 `max(x)` 后，最大的指数是 `exp(0) = 1`，不可能溢出。求和中至少有一项是 1，所以和至少是 1，而 `log(1) = 0`，不可能下溢到 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    （加减 c）
= log(sum(exp(x_i - c) * exp(c)))               （exp(a+b) = exp(a)*exp(b)）
= log(exp(c) * sum(exp(x_i - c)))               （提取公因子 exp(c)）
= c + log(sum(exp(x_i - c)))                    （log(a*b) = log(a) + log(b)）
```

令 `c = max(x)` 即可消除溢出。

这个技巧在 ML 中无处不在：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型
- 变分推断

### 为什么 Softmax 需要最大值减法技巧

Softmax 将 logits 转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

不使用技巧时，logits 为 [100, 101, 102] 会导致溢出：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

这些数在 float32 中会溢出（最大约 3.4e38）吗？是的：
exp(88.7) 就已经到了 float32 的极限。
在 float32 中 exp(100) = inf。
```

使用技巧后，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率完全一样，但计算过程是安全的。这不是优化，而是保证正确性的必要措施。

### NaN 和 Inf：检测与预防

`nan`（非数值）和 `inf`（无穷大）会像病毒一样在计算中传播。梯度更新中出现一个 `nan` 就会让权重变成 `nan`，进而让后续所有输出都变成 `nan`。一步之内训练就死了。

`inf` 的出现方式：
- 对大正数调用 `exp()`
- 除以零：`1.0 / 0.0`
- `float32` 在累加中溢出

`nan` 的出现方式：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 对负数取 `sqrt()`
- 对负数取 `log()`
- 任何涉及已有 `nan` 的运算

检测方法：

```python
import math

math.isnan(x)       # 如果 x 是 nan 则返回 True
math.isinf(x)       # 如果 x 是 +inf 或 -inf 则返回 True
math.isfinite(x)    # 如果 x 既不是 nan 也不是 inf 则返回 True
```

预防策略：

1. 限制 `exp()` 的输入：`exp(clamp(x, -80, 80))`
2. 给分母加 epsilon：`x / (y + 1e-8)`
3. 给 `log()` 内部加 epsilon：`log(x + 1e-8)`
4. 使用数值稳定的实现（log-sum-exp、稳定版 softmax）
5. 梯度裁剪防止权重爆炸
6. 调试时在每次前向传播后检查 `nan`/`inf`

### 数值梯度检验

解析梯度（来自反向传播）可能存在 bug。数值梯度检验通过有限差分计算梯度来验证解析梯度的正确性。

中心差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这是 O(h^2) 精度的，比前向差分 `(f(x+h) - f(x)) / h`（只有 O(h) 精度）好得多。

h 的选择：太大则近似不准确，太小则灾难性抵消会破坏结果。典型取值为 `h = 1e-5` 到 `1e-7`。

检验方法：计算解析梯度与数值梯度之间的相对差异。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可接受，很可能正确
- relative_error > 1e-3：有问题
- relative_error > 1：梯度完全错误

实现新的层或损失函数时一定要检验梯度。PyTorch 提供了 `torch.autograd.gradcheck()` 来完成这个工作。

### 混合精度训练

现代 GPU 拥有专用硬件（Tensor Cores），能够以比 float32 快 2-8 倍的速度计算 float16 矩阵乘法。混合精度训练（Mixed Precision Training）利用了这一点：

```
1. 维护 float32 的权重主副本
2. 前向传播使用 float16（快速）
3. 以 float32 计算损失（防止溢出）
4. 反向传播使用 float16（快速）
5. 将梯度转换为 float32
6. 更新 float32 的主权重
```

纯 float16 训练的问题：梯度通常非常小（1e-8 或更小）。float16 会将低于约 6e-8 的值下溢为零。你的模型停止学习，因为所有梯度更新都是零。

修复方法是损失缩放（Loss Scaling）：

```
1. 将 loss 乘以一个大的缩放因子（如 1024）
2. 反向传播计算 (loss * 1024) 的梯度
3. 所有梯度都大了 1024 倍（被推到 float16 下溢边界之上）
4. 更新权重前将梯度除以 1024
5. 净效果：相同的更新，但没有下溢
```

动态损失缩放会自动调整缩放因子。从一个大值（65536）开始。如果梯度溢出为 `inf`，就减半。如果 N 步没有溢出，就加倍。

### bfloat16 与 float16：为什么训练时 bfloat16 更胜一筹

```
float16:   [1 符号位] [5 指数位]  [10 尾数位]
bfloat16:  [1 符号位] [8 指数位]  [7 尾数位]
```

float16 有更高的精度（10 位尾数 vs 7 位）但范围有限（最大约 65,504）。bfloat16 精度较低，但与 float32 具有相同的范围（最大约 3.4e38）。

对于训练神经网络来说：

- 训练过程中的激活值和 logits 经常在峰值时超过 65,504。float16 溢出；bfloat16 则能应对。
- 使用 float16 需要损失缩放，但使用 bfloat16 通常不需要，因为它的范围覆盖了梯度量级的整个频谱。
- bfloat16 是 float32 的简单截断：丢掉尾数的低 16 位。转换简单直接，且在指数部分无损。

float16 适用于推理，此时数值有界且精度更重要。bfloat16 适用于训练，此时范围更重要。这就是为什么 TPU 和现代 NVIDIA GPU（A100、H100）都原生支持 bfloat16。

### 梯度裁剪

梯度爆炸发生在梯度通过多层指数级增长时（在 RNN、深层网络和 Transformer 中很常见）。一个大的梯度就能在一步内破坏所有权重。

两种裁剪方式：

**按值裁剪：** 独立地限制每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单，但可能改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保持梯度的方向不变。这就是 `torch.nn.utils.clip_grad_norm_()` 所做的事情，也是标准选择。

典型取值：Transformer 使用 `max_norm=1.0`，强化学习使用 `max_norm=0.5`，简单网络使用 `max_norm=5.0`。

梯度裁剪不是权宜之计，而是安全机制。没有它，一个异常的 batch 就能产生足够大的梯度，毁掉数周的训练成果。

### 归一化层作为数值稳定器

批归一化（Batch Normalization）、层归一化（Layer Normalization）和 RMS 归一化（RMS Normalization）通常被视为帮助训练收敛的正则化手段。它们同时也是数值稳定器。

没有归一化时，激活值可能在各层中指数级增长或缩小：

```
第 1 层：值在 [0, 1]
第 5 层：值在 [0, 100]
第 10 层：值在 [0, 10,000]
第 50 层：值在 [0, inf]
```

归一化在每一层重新居中并缩放激活值：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常为 1e-5）防止在所有激活值相同时除以零。可学习参数 `gamma` 和 `beta` 让网络恢复到它需要的任意尺度。

这使得数值在整个网络中保持在安全范围内，既防止前向传播中的溢出，也防止反向传播中的梯度爆炸。

### 常见的 ML 数值 Bug

**Bug：训练几个 epoch 后 loss 变成 NaN。**
原因：logits 增长过大，softmax 溢出。或者学习率过高，权重发散。
修复：使用稳定版 softmax（最大值减法），降低学习率，添加梯度裁剪。

**Bug：Loss 卡在 log(num_classes)。**
原因：模型输出接近均匀概率。通常意味着梯度消失或模型根本没有学习。
修复：检查数据标签是否正确，验证损失函数，检查是否存在死 ReLU。

**Bug：验证精度比预期低 1-3%。**
原因：混合精度训练没有正确使用损失缩放。梯度下溢静默地将小更新归零。
修复：启用动态损失缩放，或切换到 bfloat16。

**Bug：某些层的梯度范数为 0.0。**
原因：死 ReLU 神经元（所有输入为负），或 float16 下溢。
修复：使用 LeakyReLU 或 GELU，使用梯度缩放，检查权重初始化。

**Bug：模型在一个 GPU 上能工作，但在另一个 GPU 上给出不同结果。**
原因：非确定性的浮点累加顺序。GPU 并行规约在不同硬件上以不同顺序求和，而浮点加法不满足结合律。
修复：接受微小差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受速度损失。

**Bug：损失计算中 `exp()` 返回 `inf`。**
原因：原始 logits 直接传给 `exp()`，没有使用最大值减法技巧。
修复：使用 `torch.nn.functional.log_softmax()`，它内部实现了 log-sum-exp。

**Bug：从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示低于 6e-8 的梯度量级或高于 65,504 的激活值。
修复：使用带损失缩放的混合精度训练（AMP），或改用 bfloat16。

## 动手构建

### 第一步：展示浮点精度限制

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 第二步：实现朴素版 vs 稳定版 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) 会返回 [nan, nan, nan]
```

### 第三步：实现稳定版 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) 返回 inf
```

### 第四步：实现稳定版交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 第五步：梯度检验

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 实际应用

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整实现及所有边界情况的演示请参见 `code/numerical.py`。

## 交付成果

本课产出：
- `code/numerical.py`，包含稳定版 softmax、log-sum-exp、交叉熵、梯度检验和混合精度模拟
- `outputs/prompt-numerical-debugger.md`，用于诊断训练中的 NaN/Inf 和数值问题

这些稳定实现将在第三阶段构建训练循环和第四阶段实现注意力机制时再次出现。

## 练习

1. **灾难性抵消。** 在 float32 下用朴素公式 `E[x^2] - E[x]^2` 计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后用 Welford 在线算法计算。将误差与真实方差（0.6667）进行对比。

2. **精度探索。** 找出最小的正 float32 值 `x`，使得在 Python 中 `1.0 + x == 1.0`。这就是机器精度（Machine Epsilon）。验证它是否与 `numpy.finfo(numpy.float32).eps` 一致。

3. **Log-sum-exp 边界情况。** 用以下输入测试你的 `logsumexp_stable` 函数：(a) 所有值相等，(b) 一个值远大于其他值，(c) 所有值都非常负（-1000）。验证它在朴素版本失败的地方能给出正确结果。

4. **对神经网络层做梯度检验。** 实现一个单线性层 `y = Wx + b` 及其解析反向传播。使用 `numerical_gradient` 对一个 3x2 的权重矩阵验证正确性。

5. **损失缩放实验。** 模拟 float16 训练：创建 [1e-9, 1e-3] 范围内的随机梯度，转换为 float16，测量有多少比例变为零。然后应用损失缩放（乘以 1024），转换为 float16，再缩放回来，测量零的比例。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| IEEE 754 | "浮点标准" | 定义二进制浮点格式、舍入规则和特殊值（inf、nan）的国际标准。每个现代 CPU 和 GPU 都实现了它。 |
| 机器精度 | "精度极限" | 在给定浮点格式中，使 1.0 + e != 1.0 的最小值 e。对 float32 来说，约为 1.19e-7。 |
| 灾难性抵消 | "减法导致的精度损失" | 当对两个近似相等的浮点数做减法时，有效数字抵消，舍入噪声主导了结果。 |
| 溢出 | "数太大了" | 结果超过了最大可表示值，变成 inf。exp(89) 在 float32 中溢出。 |
| 下溢 | "数太小了" | 结果比最小可表示正数更接近零，变成 0.0。exp(-104) 在 float32 中下溢。 |
| Log-sum-exp 技巧 | "先减最大值" | 通过提取 exp(max(x)) 来计算 log(sum(exp(x)))，以防止溢出和下溢。用于 softmax、交叉熵和对数概率运算。 |
| 稳定版 softmax | "不会爆炸的 softmax" | 在取指数前减去 max(logits)。数值结果完全相同，但不可能溢出。 |
| 梯度检验 | "验证你的反向传播" | 将反向传播的解析梯度与有限差分的数值梯度进行比较，以发现实现中的 bug。 |
| 混合精度 | "前向 float16，反向 float32" | 在速度关键的运算中使用低精度浮点，在数值敏感的运算中使用高精度浮点。典型加速 2-3 倍。 |
| 损失缩放 | "防止梯度下溢" | 在反向传播前将 loss 乘以一个大常数，使梯度保持在 float16 可表示的范围内，更新权重前再除以相同的常数。 |
| bfloat16 | "Brain 浮点数" | Google 的 16 位格式，有 8 位指数（与 float32 相同范围）和 7 位尾数（精度低于 float16）。训练首选。 |
| 梯度裁剪 | "限制梯度范数" | 缩放梯度向量使其范数不超过阈值。防止梯度爆炸破坏权重。 |
| NaN | "非数值" | 来自未定义运算（0/0、inf-inf、sqrt(-1)）的特殊浮点值。会传播到后续所有运算。 |
| Inf | "无穷大" | 来自溢出或除以零的特殊浮点值。可以组合产生 NaN（inf - inf、inf * 0）。 |
| 数值梯度 | "暴力求导" | 通过计算 f(x+h) 和 f(x-h) 再除以 2h 来近似导数。慢但可靠，适合验证。 |

## 延伸阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 权威参考文献，内容密集但全面
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- NVIDIA 引入 float16 训练损失缩放的论文
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- PyTorch 混合精度实用指南
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- Google 为 TPU 选择此格式的原因
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 减少浮点求和舍入误差的算法
