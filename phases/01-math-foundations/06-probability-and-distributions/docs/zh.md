# 概率与分布

> 概率是 AI 用来表达不确定性的语言。

**类型：** 学习
**语言：** Python
**前置课程：** 第 1 阶段，第 01-04 课
**时间：** 约 75 分钟

## 学习目标

- 从零实现伯努利分布、分类分布、泊松分布、均匀分布和正态分布的 PMF 和 PDF
- 计算期望值和方差，并利用中心极限定理解释为什么高斯分布无处不在
- 构建 softmax 和 log-softmax 函数，并使用数值稳定性技巧（减去最大 logit 值）
- 从 logits 计算交叉熵损失，并将其与负对数似然联系起来

## 问题

一个分类器输出 `[0.03, 0.91, 0.06]`。一个语言模型从 50,000 个候选词中选择下一个词。一个扩散模型通过从学习到的分布中采样来生成图像。这些都是概率在发挥作用。

模型做出的每一个预测都是一个概率分布（Probability Distribution）。每一个损失函数都在衡量预测分布与真实分布之间的距离。每一个训练步骤都在调整参数，使一个分布看起来更像另一个。没有概率知识，你无法读懂一篇 ML 论文，无法调试一个模型，也无法理解为什么你的训练损失变成了 NaN。

## 概念

### 事件、样本空间和概率

样本空间（Sample Space）S 是所有可能结果的集合。事件是样本空间的一个子集。概率将事件映射到 0 和 1 之间的数字。

```
抛硬币：
  S = {正面, 反面}
  P(正面) = 0.5,  P(反面) = 0.5

掷一次骰子：
  S = {1, 2, 3, 4, 5, 6}
  P(偶数) = P({2, 4, 6}) = 3/6 = 0.5
```

三条公理定义了所有概率：
1. P(A) >= 0，对于任意事件 A
2. P(S) = 1（总会有某件事发生）
3. P(A 或 B) = P(A) + P(B)，当 A 和 B 不能同时发生时

其他一切（贝叶斯定理、期望、分布）都从这三条规则推导而来。

### 条件概率与独立性

P(A|B) 是在 B 已经发生的条件下 A 发生的概率。

```
P(A|B) = P(A 且 B) / P(B)

例子：一副扑克牌
  P(国王 | 花牌) = P(国王 且 花牌) / P(花牌)
                 = (4/52) / (12/52)
                 = 4/12 = 1/3
```

两个事件是独立的，当知道其中一个对另一个没有任何信息：

```
独立：    P(A|B) = P(A)
等价于：  P(A 且 B) = P(A) * P(B)
```

抛硬币是独立的。不放回抽牌则不是。

### 概率质量函数 vs 概率密度函数

离散随机变量有概率质量函数（PMF，Probability Mass Function）。每个结果都有一个可以直接读取的特定概率。

```
PMF: P(X = k)

公平骰子：
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  所有概率之和 = 1
```

连续随机变量有概率密度函数（PDF，Probability Density Function）。单个点的密度值不是概率。概率来自于对一个区间上的密度进行积分。

```
PDF: f(x)

P(a <= X <= b) = 从 a 到 b 对 f(x) 积分

f(x) 可以大于 1（密度，不是概率）
从 -inf 到 +inf 对 f(x) dx 积分 = 1
```

这个区别在 ML 中很重要。分类输出是 PMF（离散选择）。VAE 的潜在空间使用 PDF（连续分布）。

### 常见分布

**伯努利分布（Bernoulli）：** 一次试验，两个结果。用于二分类。

```
P(X = 1) = p
P(X = 0) = 1 - p
均值 = p，方差 = p(1-p)
```

**分类分布（Categorical）：** 一次试验，k 个结果。用于多分类（softmax 输出）。

```
P(X = i) = p_i，其中 p_i 之和 = 1
例子：P(猫) = 0.7，P(狗) = 0.2，P(鸟) = 0.1
```

**均匀分布（Uniform）：** 所有结果的概率相同。用于随机初始化。

```
离散：P(X = k) = 1/n，k 取 {1, ..., n}
连续：f(x) = 1/(b-a)，x 在 [a, b] 区间内
```

**正态分布（Normal / Gaussian）：** 钟形曲线。由均值 mu 和方差 sigma^2 参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

标准正态分布：mu = 0, sigma = 1
  68% 的数据在 1 个 sigma 范围内
  95% 在 2 个 sigma 范围内
  99.7% 在 3 个 sigma 范围内
```

**泊松分布（Poisson）：** 固定区间内稀有事件的计数。用于事件率建模。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
均值 = lambda，方差 = lambda
```

### 期望值和方差

期望值（Expected Value）是结果的加权平均。

```
离散：  E[X] = 所有 x_i * P(X = x_i) 之和
连续：  E[X] = x * f(x) dx 的积分
```

方差（Variance）衡量围绕均值的离散程度。

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
标准差 = sqrt(Var(X))
```

在 ML 中，期望值以损失函数的形式出现（数据分布上的平均损失）。方差告诉你模型的稳定性。梯度的高方差意味着训练过程噪声大。

### 联合分布和边缘分布

联合分布（Joint Distribution）P(X, Y) 同时描述两个随机变量。

联合 PMF 示例（X = 天气，Y = 是否带伞）：

| | Y=0（不带伞） | Y=1（带伞） | 边缘分布 P(X) |
|---|---|---|---|
| X=0（晴天） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（雨天） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘分布 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布（Marginal Distribution）将另一个变量求和消去：

```
P(X = x) = 对所有 y 求和 P(X = x, Y = y)
```

上表中的行合计和列合计就是边缘分布。

### 为什么正态分布无处不在

中心极限定理（Central Limit Theorem）：许多独立随机变量的和（或平均值）收敛于正态分布，与原始分布无关。

```
掷 1 次骰子：均匀分布（平坦的）
2 次骰子取平均：三角分布（有峰值）
30 次骰子取平均：近乎完美的钟形曲线

对任何初始分布都成立。
```

这就是为什么：
- 测量误差近似正态分布（许多微小的独立来源）
- 神经网络的权重初始化使用正态分布
- SGD 中的梯度噪声近似正态分布（许多样本梯度之和）
- 正态分布是在给定均值和方差下的最大熵分布

### 对数概率

原始概率会导致数值问题。许多小概率相乘很快就会下溢到零。

```
P(句子) = P(词1) * P(词2) * ... * P(词n)
        = 0.01 * 0.003 * 0.02 * ...
        -> 0.0（大约 30 项后下溢）
```

对数概率（Log Probability）解决了这个问题。乘法变成加法。

```
log P(句子) = log P(词1) + log P(词2) + ... + log P(词n)
            = -4.6 + -5.8 + -3.9 + ...
            -> 有限数值（不会下溢）
```

规则：
- log(a * b) = log(a) + log(b)
- 对数概率总是 <= 0（因为 0 < P <= 1）
- 越负表示越不可能
- 交叉熵损失就是正确类别的负对数概率

### Softmax 作为概率分布

神经网络输出原始分数（logits）。Softmax 将它们转换为有效的概率分布。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) 对所有 j)

性质：
  - 所有输出在 (0, 1) 范围内
  - 所有输出之和为 1
  - 保持输入的相对排序
  - exp() 放大 logits 之间的差异
```

Softmax 技巧：在指数运算前减去最大 logit 值，以防止溢出。

```
z = [100, 101, 102]
exp(102) = 溢出

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1（安全）

结果相同，不会溢出。
```

Log-softmax 将 softmax 和 log 结合起来以保证数值稳定性。PyTorch 在计算交叉熵损失时内部使用了这种方法。

### 采样

采样（Sampling）是从分布中随机抽取值。在 ML 中：
- Dropout 随机采样决定哪些神经元置零
- 数据增强采样随机变换
- 语言模型从预测分布中采样下一个 token
- 扩散模型采样噪声并逐步去噪

从任意分布中采样需要逆变换采样、拒绝采样或重参数化技巧（用于 VAE）等技术。

## 构建它

### 第 1 步：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 第 2 步：从零实现 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 第 3 步：期望值和方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 第 4 步：从分布中采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 第 5 步：Softmax 和对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 第 6 步：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 第 7 步：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

所有完整实现及可视化代码在 `code/probability.py` 中。

## 使用它

使用 NumPy 和 SciPy，上面的所有内容都可以用一行代码完成：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你已经从零构建了这些。现在你知道库函数在背后做了什么。

## 练习

1. 为指数分布实现逆变换采样。通过采样 10,000 个值并将直方图与真实 PDF 对比来验证。

2. 为两个不均匀骰子构建联合分布表。计算边缘分布，并检验这两个骰子是否独立。

3. 计算一个 5 分类器的交叉熵损失，该分类器在正确类别为索引 3 时输出 logits `[2.0, 0.5, -1.0, 3.0, 0.1]`。然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证你的答案。

4. 编写一个函数，输入对数概率列表，返回最可能的序列、总对数概率和等价的原始概率。用一个 50 个词的句子来测试，其中每个词的概率为 0.01。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| 样本空间（Sample Space） | "所有可能性" | 一次实验中所有可能结果的集合 S |
| PMF（概率质量函数） | "概率函数" | 给出每个离散结果的精确概率的函数，总和为 1 |
| PDF（概率密度函数） | "概率曲线" | 连续变量的密度函数。对一个区间积分才能得到概率 |
| 条件概率（Conditional Probability） | "在某条件下的概率" | P(A\|B) = P(A 且 B) / P(B)。贝叶斯思维和贝叶斯定理的基础 |
| 独立性（Independence） | "它们互不影响" | P(A 且 B) = P(A) * P(B)。知道一个事件不会告诉你另一个的任何信息 |
| 期望值（Expected Value） | "平均值" | 所有结果的概率加权和。损失函数就是一个期望值 |
| 方差（Variance） | "有多分散" | 与均值的期望平方偏差。高方差 = 噪声大、估计不稳定 |
| 正态分布（Normal Distribution） | "钟形曲线" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))。由于中心极限定理而无处不在 |
| 中心极限定理（Central Limit Theorem） | "均值趋向正态" | 许多独立样本的均值收敛于正态分布，与原始分布无关 |
| 联合分布（Joint Distribution） | "两个变量在一起" | P(X, Y) 描述 X 和 Y 每种组合结果的概率 |
| 边缘分布（Marginal Distribution） | "把另一个变量求和掉" | P(X) = sum_y P(X, Y)。从联合分布中恢复单个变量的分布 |
| 对数概率（Log Probability） | "概率的对数" | log P(x)。将乘法转化为加法，防止长序列中的数值下溢 |
| Softmax | "将分数转化为概率" | softmax(z_i) = exp(z_i) / sum(exp(z_j))。将实值 logits 映射为有效的概率分布 |
| 交叉熵（Cross-entropy） | "损失函数" | -sum(p_true * log(p_predicted))。衡量两个分布之间的差异。越低越好 |
| Logits | "模型原始输出" | softmax 之前的未归一化分数。得名于 logistic 函数 |
| 采样（Sampling） | "随机抽取值" | 按照概率分布生成值。模型生成输出的方式 |

## 延伸阅读

- [3Blue1Brown: 什么是中心极限定理？](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 为什么均值趋向正态的可视化证明
- [Stanford CS229 概率论复习](https://cs229.stanford.edu/section/cs229-prob.pdf) - 涵盖本课内容及更多的简洁参考资料
- [Log-Sum-Exp 技巧](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 为什么数值稳定性重要以及如何实现
