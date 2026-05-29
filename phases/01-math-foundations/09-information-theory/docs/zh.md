# 信息论

> 信息论度量的是意外程度。损失函数就是建立在它之上的。

**类型：** 学习
**语言：** Python
**前置知识：** 第1阶段，第06课（概率论）
**时长：** 约60分钟

## 学习目标

- 从零实现熵（Entropy）、交叉熵（Cross-Entropy）和 KL 散度（KL Divergence），并解释它们之间的关系
- 推导为什么最小化交叉熵损失等价于最大化对数似然
- 计算特征与目标之间的互信息（Mutual Information），用于特征重要性排序
- 解释困惑度（Perplexity）作为语言模型在选择时的等效词汇量大小

## 问题引入

你在训练每一个分类模型时都会调用 `CrossEntropyLoss()`。你在每篇语言模型论文中都会看到"困惑度"。你在 VAE、知识蒸馏和 RLHF 中都会读到 KL 散度。这些并不是孤立的概念，它们其实是同一个思想穿着不同的外衣。

信息论为你提供了一套推理不确定性、压缩和预测的语言。Claude Shannon 在1948年为了解决通信问题而发明了它。事实证明，训练一个神经网络本身就是一个通信问题：模型试图通过由学习到的权重构成的噪声信道，传输正确的标签。

本课将从零推导每一个公式，让你看到它们从何而来、为何有效。

## 核心概念

### 信息量（意外程度）

当不太可能的事件发生时，它携带了更多的信息。抛硬币正面朝上？不意外。中彩票？非常意外。

概率为 p 的事件的信息量为：

```
I(x) = -log(p(x))
```

使用以2为底的对数，单位是比特（bit）。使用自然对数，单位是奈特（nat）。思想相同，单位不同。

```
事件                概率        意外程度（比特）
公平硬币正面朝上     0.5            1.0
掷出6点             0.167          2.58
千分之一的事件       0.001          9.97
确定事件             1.0            0.0
```

确定事件携带零信息——你本来就知道它会发生。

### 熵（平均意外程度）

熵是一个分布中所有可能结果的期望意外程度。

```
H(P) = -sum( p(x) * log(p(x)) )  对所有 x
```

一枚公平硬币对二元变量具有最大熵：1比特。一枚有偏硬币（99%正面）的熵很低：0.08比特。你已经知道结果会是什么，所以每次抛掷几乎不会告诉你任何新信息。

```
公平硬币：  H = -(0.5 * log2(0.5) + 0.5 * log2(0.5)) = 1.0 比特
有偏硬币：  H = -(0.99 * log2(0.99) + 0.01 * log2(0.01)) = 0.08 比特
```

熵度量的是分布中不可约减的不确定性。你无法压缩到低于它的程度。

### 交叉熵（你每天都在用的损失函数）

交叉熵度量的是：当你用分布 Q 来编码实际来自分布 P 的事件时，平均会产生多少意外。

```
H(P, Q) = -sum( p(x) * log(q(x)) )  对所有 x
```

P 是真实分布（标签），Q 是你的模型预测。如果 Q 完美匹配 P，交叉熵就等于熵。任何不匹配都会使它变大。

在分类任务中，P 是一个独热向量（one-hot vector）——真实类别的概率为1，其他都为0。这将交叉熵简化为：

```
H(P, Q) = -log(q(真实类别))
```

这就是分类任务的全部交叉熵损失公式。目标就是最大化模型对正确类别的预测概率。

### KL 散度（分布间的距离）

KL 散度度量的是：用 Q 代替 P 会产生多少额外的意外。

```
D_KL(P || Q) = sum( p(x) * log(p(x) / q(x)) )  对所有 x
             = H(P, Q) - H(P)
```

交叉熵等于熵加上 KL 散度。由于真实分布的熵在训练过程中是常数，因此最小化交叉熵就等同于最小化 KL 散度。你正在将模型的分布推向真实分布。

KL 散度不是对称的：D_KL(P || Q) != D_KL(Q || P)。它不是一个真正的距离度量。

### 互信息

互信息度量的是：知道一个变量能告诉你关于另一个变量多少信息。

```
I(X; Y) = H(X) - H(X|Y)
        = H(X) + H(Y) - H(X, Y)
```

如果 X 和 Y 独立，互信息为零——知道一个对另一个毫无帮助。如果它们完全相关，互信息等于任一变量的熵。

在特征选择中，特征与目标之间的高互信息意味着该特征有用。低互信息意味着它是噪声。

### 条件熵

H(Y|X) 度量的是：在观测到 X 之后，对 Y 还剩多少不确定性。

```
H(Y|X) = H(X,Y) - H(X)
```

两个极端情况：
- 如果 X 完全决定了 Y，则 H(Y|X) = 0。知道 X 就消除了关于 Y 的所有不确定性。例如：X = 摄氏温度，Y = 华氏温度。
- 如果 X 对 Y 毫无帮助，则 H(Y|X) = H(Y)。知道 X 完全不能减少你对 Y 的不确定性。例如：X = 抛硬币结果，Y = 明天的天气。

条件熵始终非负，且不超过 H(Y)：

```
0 <= H(Y|X) <= H(Y)
```

在机器学习中，条件熵出现在决策树中。在每次分裂时，算法选择能最小化 H(Y|X) 的特征 X——即最能消除标签 Y 不确定性的特征。

### 联合熵

H(X,Y) 是 X 和 Y 联合分布的熵。

```
H(X,Y) = -sum sum p(x,y) * log(p(x,y))   对所有 x, y
```

关键性质：

```
H(X,Y) <= H(X) + H(Y)
```

当 X 和 Y 独立时等号成立。如果它们共享信息，联合熵就小于各自熵的和。"缺失"的那部分熵恰好就是互信息。

```mermaid
graph TD
    subgraph "信息维恩图"
        direction LR
        HX["H(X)"]
        HY["H(Y)"]
        MI["I(X;Y)<br/>互信息"]
        HXgY["H(X|Y)<br/>= H(X) - I(X;Y)"]
        HYgX["H(Y|X)<br/>= H(Y) - I(X;Y)"]
        HXY["H(X,Y) = H(X) + H(Y) - I(X;Y)"]
    end

    HXgY --- MI
    MI --- HYgX
    HX -.- HXgY
    HX -.- MI
    HY -.- MI
    HY -.- HYgX
    HXY -.- HXgY
    HXY -.- MI
    HXY -.- HYgX
```

它们之间的关系：
- H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
- I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
- H(X,Y) = H(X) + H(Y) - I(X;Y)

### 互信息（深入探讨）

互信息 I(X;Y) 量化了知道一个变量能多大程度地减少对另一个变量的不确定性。

```
I(X;Y) = H(X) - H(X|Y)
       = H(Y) - H(Y|X)
       = H(X) + H(Y) - H(X,Y)
       = sum sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
```

性质：
- I(X;Y) >= 0 恒成立。观测某事物永远不会让你丢失信息。
- I(X;Y) = 0 当且仅当 X 和 Y 独立。
- I(X;Y) = I(Y;X)。它是对称的，这一点与 KL 散度不同。
- I(X;X) = H(X)。一个变量与自身共享其全部信息。

**互信息用于特征选择。** 在机器学习中，你希望选择对目标有信息量的特征。互信息为你提供了一种有原理依据的特征排序方法：

1. 对每个特征 X_i，计算 I(X_i; Y)，其中 Y 是目标变量。
2. 按互信息得分对特征排序。
3. 保留前 k 个特征。

这适用于特征和目标之间的任何关系——线性、非线性、单调或非单调。相关系数只能捕捉线性关系，而互信息能捕捉一切。

| 方法 | 能检测的关系 | 计算成本 | 能处理类别变量？ |
|--------|---------|-------------------|---------------------|
| Pearson 相关系数 | 线性关系 | O(n) | 否 |
| Spearman 相关系数 | 单调关系 | O(n log n) | 否 |
| 互信息 | 任何统计依赖关系 | O(n log n)（分箱法） | 是 |

### 标签平滑与交叉熵

标准分类使用硬目标：[0, 0, 1, 0]。真实类别的概率为1，其他为0。标签平滑（Label Smoothing）将它们替换为软目标：

```
soft_target = (1 - epsilon) * hard_target + epsilon / num_classes
```

当 epsilon = 0.1，有4个类别时：
- 硬目标：[0, 0, 1, 0]
- 软目标：[0.025, 0.025, 0.925, 0.025]

从信息论的角度来看，标签平滑增加了目标分布的熵。硬独热目标的熵为0——没有不确定性。软目标具有正的熵。

为什么这有帮助：
- 防止模型将 logits 推向极端值（在交叉熵下，需要无穷大的 logits 才能完美匹配独热目标）
- 起到正则化的作用：模型不能100%确信
- 改善校准：预测概率更好地反映真实的不确定性
- 缩小训练和推理行为之间的差距

带标签平滑的交叉熵损失变为：

```
L = (1 - epsilon) * CE(hard_target, prediction) + epsilon * H_uniform(prediction)
```

第二项惩罚远离均匀分布的预测——这是对置信度的直接正则化。

### 为什么交叉熵是分类的首选损失函数

三个视角，同一个结论。

**信息论视角。** 交叉熵度量的是用模型分布代替真实分布时浪费了多少比特。最小化它使你的模型成为现实最高效的编码器。

**最大似然视角。** 对于 N 个训练样本及其真实类别 y_i：

```
似然         = product( q(y_i) )
对数似然     = sum( log(q(y_i)) )
负对数似然   = -sum( log(q(y_i)) )
```

最后一行就是交叉熵损失。最小化交叉熵 = 最大化训练数据在你的模型下的似然。

**梯度视角。** 交叉熵对 logits 的梯度简单地等于（预测值 - 真实值）。简洁、稳定、计算快速。这就是它与 softmax 完美配合的原因。

### 比特与奈特

唯一的区别在于对数的底数。

```
以2为底   -> 比特（bits）    （信息论传统）
以e为底   -> 奈特（nats）    （机器学习惯例）
以10为底  -> 哈特利（hartleys）（很少使用）
```

1 奈特 = 1/ln(2) 比特 = 1.4427 比特。PyTorch 和 TensorFlow 默认使用自然对数（奈特）。

### 困惑度

困惑度是交叉熵的指数。它告诉你模型在选择时等效于从多少个等概率选项中进行选择。

```
困惑度 = 2^H(P,Q)   （使用比特时）
困惑度 = e^H(P,Q)   （使用奈特时）
```

一个困惑度为50的语言模型，平均而言，就像从50个可能的下一个 token 中均匀随机选择一样迷茫。越低越好。

GPT-2 在常见基准测试上的困惑度约为30。现代模型在覆盖良好的领域中可以达到个位数。

## 动手实现

### 第1步：信息量与熵

```python
import math

def information_content(p, base=2):
    if p <= 0 or p > 1:
        return float('inf') if p <= 0 else 0.0
    return -math.log(p) / math.log(base)

def entropy(probs, base=2):
    return sum(
        p * information_content(p, base)
        for p in probs if p > 0
    )

fair_coin = [0.5, 0.5]
biased_coin = [0.99, 0.01]
fair_die = [1/6] * 6

print(f"Fair coin entropy:   {entropy(fair_coin):.4f} bits")
print(f"Biased coin entropy: {entropy(biased_coin):.4f} bits")
print(f"Fair die entropy:    {entropy(fair_die):.4f} bits")
```

### 第2步：交叉熵与 KL 散度

```python
def cross_entropy(p, q, base=2):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * (-math.log(qi) / math.log(base))
    return total

def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)

true_dist = [0.7, 0.2, 0.1]
good_model = [0.6, 0.25, 0.15]
bad_model = [0.1, 0.1, 0.8]

print(f"Entropy of true dist:     {entropy(true_dist):.4f} bits")
print(f"CE (good model):          {cross_entropy(true_dist, good_model):.4f} bits")
print(f"CE (bad model):           {cross_entropy(true_dist, bad_model):.4f} bits")
print(f"KL divergence (good):     {kl_divergence(true_dist, good_model):.4f} bits")
print(f"KL divergence (bad):      {kl_divergence(true_dist, bad_model):.4f} bits")
```

### 第3步：交叉熵作为分类损失

```python
def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def cross_entropy_loss(true_class, logits):
    probs = softmax(logits)
    return -math.log(probs[true_class])

logits = [2.0, 1.0, 0.1]
true_class = 0

probs = softmax(logits)
loss = cross_entropy_loss(true_class, logits)

print(f"Logits:      {logits}")
print(f"Softmax:     {[f'{p:.4f}' for p in probs]}")
print(f"True class:  {true_class}")
print(f"Loss:        {loss:.4f} nats")
print(f"Perplexity:  {math.exp(loss):.2f}")
```

### 第4步：交叉熵等于负对数似然

```python
import random

random.seed(42)

n_samples = 1000
n_classes = 3
true_labels = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
model_logits = [[random.gauss(0, 1) for _ in range(n_classes)] for _ in range(n_samples)]

ce_loss = sum(
    cross_entropy_loss(label, logits)
    for label, logits in zip(true_labels, model_logits)
) / n_samples

nll = -sum(
    math.log(softmax(logits)[label])
    for label, logits in zip(true_labels, model_logits)
) / n_samples

print(f"Cross-entropy loss:      {ce_loss:.6f}")
print(f"Negative log-likelihood: {nll:.6f}")
print(f"Difference:              {abs(ce_loss - nll):.2e}")
```

### 第5步：互信息

```python
def mutual_information(joint_probs, base=2):
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    margin_x = [sum(joint_probs[i][j] for j in range(cols)) for i in range(rows)]
    margin_y = [sum(joint_probs[i][j] for i in range(rows)) for j in range(cols)]

    mi = 0.0
    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            if pxy > 0:
                mi += pxy * math.log(pxy / (margin_x[i] * margin_y[j])) / math.log(base)
    return mi

independent = [[0.25, 0.25], [0.25, 0.25]]
dependent = [[0.45, 0.05], [0.05, 0.45]]

print(f"MI (independent): {mutual_information(independent):.4f} bits")
print(f"MI (dependent):   {mutual_information(dependent):.4f} bits")
```

## 实际应用

使用 NumPy 实现相同的概念，这才是你在实际工作中会使用的方式：

```python
import numpy as np

def np_entropy(p):
    p = np.asarray(p, dtype=float)
    mask = p > 0
    result = np.zeros_like(p)
    result[mask] = p[mask] * np.log(p[mask])
    return -result.sum()

def np_cross_entropy(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    return -(p[mask] * np.log(q[mask])).sum()

def np_kl_divergence(p, q):
    return np_cross_entropy(p, q) - np_entropy(p)

true = np.array([0.7, 0.2, 0.1])
pred = np.array([0.6, 0.25, 0.15])
print(f"Entropy:    {np_entropy(true):.4f} nats")
print(f"Cross-ent:  {np_cross_entropy(true, pred):.4f} nats")
print(f"KL div:     {np_kl_divergence(true, pred):.4f} nats")
```

你从零构建了 `torch.nn.CrossEntropyLoss()` 内部所做的事情。现在你明白了为什么训练过程中损失会下降：你的模型预测分布正在不断接近真实分布，以浪费的信息奈特数来衡量。

## 练习

1. 假设英文字母表是均匀分布的（26个字母），计算其熵。然后用实际的字母频率来估算。哪个更高？为什么？

2. 一个模型对一个真实类别为1的样本输出 logits [5.0, 2.0, 0.5]。手动计算交叉熵损失，然后用你的 `cross_entropy_loss` 函数验证。什么样的 logits 能使损失为零？

3. 证明 KL 散度不是对称的。选择两个分布 P 和 Q，分别计算 D_KL(P || Q) 和 D_KL(Q || P)。解释为什么它们不同。

4. 构建一个函数来计算 token 预测序列的困惑度。给定一组 (真实token索引, 预测logits) 对，返回该序列的困惑度。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| 信息量 | "意外程度" | 编码一个事件所需的比特（或奈特）数：-log(p) |
| 熵 | "随机性" | 分布中所有结果的平均意外程度。度量不可约减的不确定性。 |
| 交叉熵 | "损失函数" | 用模型分布 Q 编码来自真实分布 P 的事件时的平均意外程度。 |
| KL 散度 | "分布间的距离" | 用 Q 代替 P 浪费的额外比特数。等于交叉熵减去熵。不对称。 |
| 互信息 | "X 和 Y 有多相关" | 知道 Y 后对 X 不确定性的减少量。为零意味着独立。 |
| Softmax | "将 logits 转换为概率" | 取指数然后归一化。将任意实数向量映射为有效的概率分布。 |
| 困惑度 | "模型有多迷茫" | 交叉熵的指数。模型在每一步选择时等效的词汇量大小。 |
| 比特 | "Shannon 的单位" | 以2为底的对数度量的信息。1比特能解决一次公平硬币的翻转。 |
| 奈特 | "机器学习的单位" | 以自然对数度量的信息。PyTorch 和 TensorFlow 默认使用。 |
| 负对数似然 | "NLL 损失" | 对于独热标签，与交叉熵损失完全相同。最小化它等于最大化正确预测的概率。 |

## 延伸阅读

- [Shannon 1948: A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) - 原始论文，至今仍然易读
- [Visual Information Theory (Chris Olah)](https://colah.github.io/posts/2015-09-Visual-Information/) - 关于熵和 KL 散度最好的可视化解释
- [PyTorch CrossEntropyLoss 文档](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) - 框架如何实现你刚才构建的内容
