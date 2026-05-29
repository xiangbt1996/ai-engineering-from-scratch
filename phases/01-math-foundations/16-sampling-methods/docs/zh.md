# 采样方法

> 采样是 AI 探索可能性空间的方式。

**类型：** 动手构建
**语言：** Python
**前置课程：** 第一阶段，第 06-07 课（概率、贝叶斯定理）
**时长：** 约 120 分钟

## 学习目标

- 仅使用均匀随机数，从零实现逆 CDF 采样、拒绝采样和重要性采样
- 构建用于语言模型词元生成的温度采样、top-k 采样和 top-p（核采样）
- 解释重参数化技巧（Reparameterization Trick）及其为何能在 VAE 中实现通过采样的反向传播
- 运行 Metropolis-Hastings MCMC 从未归一化的目标分布中采样

## 问题引入

一个语言模型处理完你的提示词后，产生了一个包含 50,000 个 logit 的向量——词汇表中每个词元对应一个。现在它需要选择一个。怎么选？

如果它总是选概率最高的词元，每次回复都一模一样。确定性的。无聊的。如果它均匀随机选择，输出就是乱码。答案在这两个极端之间，而这个"之间"就是由采样策略控制的。

采样不仅限于文本生成。强化学习通过采样轨迹来估计策略梯度。VAE 通过从学习到的分布中采样并通过随机性进行反向传播来学习潜在表示。扩散模型通过采样噪声并逐步去噪来生成图像。蒙特卡洛方法估计没有封闭形式解的积分。MCMC 算法探索无法枚举的高维后验分布。

每一个生成式 AI 系统都是一个采样系统。采样策略决定了输出的质量、多样性和可控性。本课从均匀随机数开始，逐步构建每一种主要的采样方法，最终实现驱动现代大语言模型和生成模型的技术。

## 核心概念

### 为什么采样很重要

采样在 AI 和机器学习中扮演四个基本角色：

**生成。** 语言模型、扩散模型和 GAN 都通过采样产生输出。采样算法直接控制创造力、连贯性和多样性。温度、top-k 和核采样是工程师每天都要调节的旋钮。

**训练。** 随机梯度下降采样小批量数据。Dropout 采样要停用的神经元。数据增强采样随机变换。重要性采样在强化学习（PPO、TRPO）中对样本进行重加权以减少梯度方差。

**估计。** 机器学习中许多量没有封闭形式解。数据分布上的期望损失、能量模型的配分函数、贝叶斯推断中的证据。蒙特卡洛估计通过对样本求平均来近似所有这些量。

**探索。** MCMC 算法在贝叶斯推断中探索后验分布。进化策略采样参数扰动。Thompson 采样在老虎机问题中平衡探索与利用。

核心挑战：你只能直接从简单分布（均匀分布、正态分布）中采样。对于其他一切，你需要一种方法将简单样本转换为来自目标分布的样本。

### 均匀随机采样

所有采样方法都从这里开始。均匀随机数生成器在 [0, 1) 范围内产生值，其中等长的子区间具有相等的概率。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    对于 0 <= a <= b <= 1

性质：
  E[U] = 0.5
  Var(U) = 1/12
```

要从 n 个元素的离散集合中均匀采样，生成 U 并返回 floor(n * U)。要从连续区间 [a, b] 采样，计算 a + (b - a) * U。

关键洞察：一个均匀随机数恰好包含了从任何分布中产生一个样本所需的随机量。技巧在于找到正确的变换。

### 逆 CDF 方法（逆变换采样）

累积分布函数（CDF）将值映射到概率：

```
F(x) = P(X <= x)

性质：
  F 是非递减的
  F(-inf) = 0
  F(+inf) = 1
  F 将实数线映射到 [0, 1]
```

逆 CDF 将概率映射回值。如果 U ~ Uniform(0, 1)，则 X = F_inverse(U) 服从目标分布。

```
算法：
  1. 生成 u ~ Uniform(0, 1)
  2. 返回 F_inverse(u)

为什么有效：
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**指数分布示例：**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

对 x 解 F(x) = u：
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

由于 (1 - U) 和 U 具有相同的分布：
  x = -ln(u) / lambda
```

当你能写出 F_inverse 的封闭形式时，这个方法完美适用。对于正态分布，没有封闭形式的逆 CDF，所以我们使用其他方法（Box-Muller 变换或数值近似）。

**离散版本：** 对于离散分布，构建 CDF 作为累积和，生成 U，然后找到累积和超过 U 的第一个索引。这就是第 06 课中 `sample_categorical` 的工作原理。

### 拒绝采样

当你无法反转 CDF 但能计算目标 PDF（可以是未归一化的）时，拒绝采样（Rejection Sampling）就派上用场了。

```
目标分布：p(x)（可以计算，可能未归一化）
提议分布：q(x)（可以从中采样）
上界：M 使得对所有 x，p(x) <= M * q(x)

算法：
  1. 采样 x ~ q(x)
  2. 采样 u ~ Uniform(0, 1)
  3. 如果 u < p(x) / (M * q(x))，接受 x
  4. 否则，拒绝并回到步骤 1

接受率 = 1/M
```

上界 M 越紧，接受率越高。在低维（1-3 维）中，拒绝采样效果良好。在高维中，接受率呈指数下降，因为提议体积的大部分被拒绝了。这就是拒绝采样的维度灾难。

**示例：从截断正态分布中采样。** 在截断范围上使用均匀提议。包络 M 是该范围内正态 PDF 的最大值。

**示例：从半圆中采样。** 在外接矩形内均匀提议。如果点落在半圆内则接受。这就是蒙特卡洛方法计算圆周率的原理：接受率等于面积比 pi/4。

### 重要性采样

有时你并不需要从目标分布 p(x) 中采样。你需要估计 p(x) 下的一个期望值，而你手头有的是来自另一个分布 q(x) 的样本。

```
目标：估计 E_p[f(x)] = integral of f(x) * p(x) dx

改写：
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

其中 w(x) = p(x) / q(x) 是重要性权重。

估计量：
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    其中 x_i ~ q(x)
```

这在强化学习中至关重要。在 PPO（近端策略优化）中，你在旧策略 pi_old 下收集轨迹，但想优化新策略 pi_new。重要性权重是 pi_new(a|s) / pi_old(a|s)。PPO 对这些权重进行裁剪，以防止新策略偏离旧策略太远。

重要性采样估计量的方差取决于 q 与 p 的相似程度。如果 q 与 p 差异很大，少数样本会获得巨大的权重并主导估计。自归一化重要性采样通过除以权重之和来缓解这个问题：

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### 蒙特卡洛估计

蒙特卡洛估计（Monte Carlo Estimation）通过对随机样本求平均来近似积分。大数定律保证了收敛性。

```
目标：估计 I = integral of g(x) dx 在域 D 上

方法：
  1. 从 D 中均匀采样 x_1, ..., x_N
  2. I ~ (D 的体积 / N) * sum(g(x_i))

误差：O(1 / sqrt(N))   与维度无关
```

误差率与维度无关。这就是为什么在高维中，基于网格的积分不可行时，蒙特卡洛方法占主导地位。

**估计圆周率：**

```
从 [-1, 1] x [-1, 1] 中均匀采样 (x, y)
计算有多少落在单位圆内：x^2 + y^2 <= 1
pi ~ 4 * (圆内个数) / (总个数)
```

**估计期望值：**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    其中 x_i ~ p(x)

样本均值收敛到真实期望。
估计量的方差 = Var(f(X)) / N
```

### 马尔可夫链蒙特卡洛（MCMC）：Metropolis-Hastings

MCMC 构建一条马尔可夫链，其平稳分布就是目标分布 p(x)。经过足够多的步骤后，链上的样本就（近似地）是来自 p(x) 的样本。

```
目标：p(x)（已知到一个归一化常数）
提议：q(x'|x)（给定当前状态如何提议下一个状态）

Metropolis-Hastings 算法：
  1. 从某个 x_0 开始
  2. 对于 t = 1, 2, ..., T：
     a. 提议 x' ~ q(x'|x_t)
     b. 计算接受比：
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. 以概率 min(1, alpha) 接受：
        - 如果 u < alpha（u ~ Uniform(0,1)）：x_{t+1} = x'
        - 否则：x_{t+1} = x_t
  3. 丢弃前 B 个样本（预热期）
  4. 返回剩余样本
```

对于对称提议（q(x'|x) = q(x|x')），比值简化为 p(x')/p(x)。这就是原始的 Metropolis 算法。

**为什么有效。** 接受规则确保了细致平衡（Detailed Balance）：处于 x 并移动到 x' 的概率等于处于 x' 并移动到 x 的概率。细致平衡意味着 p(x) 是链的平稳分布。

**实践注意事项：**
- 预热期（Burn-in）：在链达到平衡之前丢弃早期样本
- 稀疏化（Thinning）：每隔 k 个样本保留一个，以减少自相关
- 提议尺度：太小则链移动缓慢（高接受率，慢探索）；太大则大多数提议被拒绝（低接受率，原地不动）
- 在高维中使用高斯提议时，最优接受率约为 0.234

### Gibbs 采样

Gibbs 采样是 MCMC 的一个特例，用于多元分布。它不是一次在所有维度上提议移动，而是每次更新一个变量，从其条件分布中采样。

```
目标：p(x_1, x_2, ..., x_d)

算法：
  对于每次迭代 t：
    采样 x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    采样 x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    采样 x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs 采样要求你能从每个条件分布 p(x_i | x_{-i}) 中采样。这对许多模型来说是直接的：
- 贝叶斯网络：条件分布可从图结构推导
- 高斯混合模型：条件分布是高斯的
- 伊辛模型：每个自旋的条件分布只取决于其邻居

接受率始终为 1（每个提议都被接受），因为从精确的条件分布中采样自动满足细致平衡。

**局限性。** 当变量高度相关时，Gibbs 采样混合得很慢，因为一次更新一个变量无法沿对角方向大幅移动。

### 温度采样（用于大语言模型）

语言模型为词汇表中的每个词元输出 logit z_1, ..., z_V。Softmax 将其转换为概率。温度在 softmax 之前对 logit 进行缩放：

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: 标准 softmax（原始分布）
T -> 0:  argmax（确定性，总是选最高 logit）
T -> inf: 均匀分布（所有词元等概率）
T < 1.0: 使分布更尖锐（更确定，多样性更低）
T > 1.0: 使分布更平坦（更不确定，多样性更高）
```

**为什么有效。** 将 logit 除以 T < 1 会放大 logit 之间的差异。如果 z_1 = 2 且 z_2 = 1，除以 T = 0.5 得到 z_1/T = 4 和 z_2/T = 2，差距变大了。经过 softmax 后，最高 logit 的词元获得更大的概率份额。

**实践中：**
- T = 0.0：贪心解码，最适合事实问答
- T = 0.3-0.7：略有创造性，适合代码生成
- T = 0.7-1.0：平衡，适合通用对话
- T = 1.0-1.5：创意写作、头脑风暴
- T > 1.5：越来越随机，很少有用

温度不改变哪些词元是可能的，而是改变分配给每个词元的概率质量。

### Top-k 采样

Top-k 采样将候选集限制为概率最高的 k 个词元，然后重新归一化并从该受限集合中采样。

```
算法：
  1. 计算所有 V 个词元的 softmax 概率
  2. 按概率降序排列词元
  3. 仅保留前 k 个词元
  4. 重新归一化：p_i' = p_i / sum(p_j for j in top-k)
  5. 从重新归一化的分布中采样

k = 1:  贪心解码
k = V:  无过滤（标准采样）
k = 40: 典型设置，移除不太可能的词元的长尾
```

Top-k 防止模型选择词汇分布长尾中极不可能的词元（错别字、无意义文字）。问题在于：k 是固定的，与上下文无关。当模型很有信心时（一个词元有 95% 的概率），k = 40 仍然允许 39 个替代选项。当模型不确定时（概率分散在 1000 个词元上），k = 40 会截断合理的选项。

### Top-p（核采样）

Top-p 采样动态调整候选集大小。它不是保留固定数量的词元，而是保留累积概率超过 p 的最小词元集合。

```
算法：
  1. 计算所有 V 个词元的 softmax 概率
  2. 按概率降序排列词元
  3. 找到最小的 k 使得前 k 个概率之和 >= p
  4. 仅保留那些 k 个词元
  5. 重新归一化并采样

p = 0.9: 保留覆盖 90% 概率质量的词元
p = 1.0: 无过滤
p = 0.1: 非常限制性，接近贪心
```

当模型有信心时，核采样只保留少数词元（可能 2-3 个）。当模型不确定时，它保留很多（可能 200 个）。这种自适应行为是核采样通常比 top-k 产生更好文本的原因。

**常见组合：**
- 温度 0.7 + top-p 0.9：良好的通用设置
- 温度 0.0（贪心）：最适合确定性任务
- 温度 1.0 + top-k 50：Fan et al. (2018) 原始论文的设置

Top-k 和 top-p 可以结合使用。先应用 top-k，然后对剩余集合应用 top-p。

### 重参数化技巧（用于 VAE）

变分自编码器（VAE）通过将输入编码为潜在空间中的分布、从该分布中采样、然后将样本解码回来进行学习。问题是：你无法通过采样操作进行反向传播。

```
标准采样（不可微分）：
  z ~ N(mu, sigma^2)

  随机性阻断了梯度流。
  d/d_mu [从 N(mu, sigma^2) 中采样] = ???
```

重参数化技巧将随机性与参数分离：

```
重参数化采样：
  epsilon ~ N(0, 1)          （固定随机噪声，无参数）
  z = mu + sigma * epsilon   （参数的确定性函数）

  现在 z 是 mu 和 sigma 的确定性可微函数。
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  梯度可以通过 mu 和 sigma 流动。
```

这之所以有效，是因为 N(mu, sigma^2) 与 mu + sigma * N(0, 1) 具有相同的分布。关键洞察：将随机性移到一个无参数的源（epsilon），然后将样本表达为参数的可微变换。

**在 VAE 训练循环中：**
1. 编码器为每个输入输出 mu 和 log(sigma^2)
2. 采样 epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. 解码 z 以重建输入
5. 通过步骤 4、3、2、1 反向传播（因为步骤 3 是可微的，所以这是可行的）

没有重参数化技巧，VAE 无法用标准反向传播训练。这一个洞察就使 VAE 变得实用了。

### Gumbel-Softmax（可微分类别采样）

重参数化技巧适用于连续分布（高斯分布）。对于离散类别分布，我们需要不同的方法。Gumbel-Softmax 提供了类别采样的可微近似。

**Gumbel-Max 技巧（不可微分）：**

```
从对数概率 log(p_1), ..., log(p_k) 的类别分布中采样：
  1. 对每个类别采样 g_i ~ Gumbel(0, 1)
     (g = -log(-log(u)), 其中 u ~ Uniform(0, 1))
  2. 返回 argmax(log(p_i) + g_i)

这产生精确的类别采样。
```

**Gumbel-Softmax（可微近似）：**

```
用 soft softmax 替换 hard argmax：
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau（温度）控制近似程度：
  tau -> 0:   趋近于 one-hot 向量（硬类别）
  tau -> inf: 趋近于均匀分布 (1/k, 1/k, ..., 1/k)
  tau = 1.0:  软近似
```

Gumbel-Softmax 产生离散样本的连续松弛。输出是一个概率向量（soft one-hot）而不是 hard one-hot。梯度可以通过 softmax 流动。在训练的前向传播中，你可以使用"直通估计器"（Straight-through Estimator）：前向传播使用 hard argmax，反向传播使用 soft Gumbel-Softmax 梯度。

**应用：**
- VAE 中的离散潜在变量
- 神经架构搜索（选择离散操作）
- 硬注意力机制
- 离散动作的强化学习

### 分层采样

标准蒙特卡洛采样可能因为偶然在样本空间中留下空白。分层采样（Stratified Sampling）通过将空间划分为层并从每层中采样来强制均匀覆盖。

```
标准蒙特卡洛：
  从 [0, 1] 中均匀采样 N 个点
  某些区域可能有聚集，其他区域有空白

分层采样：
  将 [0, 1] 分成 N 个等份层：[0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  在每个层内均匀采样一个点
  x_i = (i + u_i) / N   其中 u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

分层采样的方差始终小于或等于标准蒙特卡洛：

```
Var(分层) <= Var(标准蒙特卡洛)

当 f(x) 平滑变化时改进最大。
对于分段常数函数，分层采样是精确的。
```

**应用：**
- 数值积分（准蒙特卡洛）
- 训练数据划分（确保每折中的类别平衡）
- 结合分层的重要性采样（两种技术的组合）
- NeRF（神经辐射场）在相机光线上使用分层采样

### 与扩散模型的联系

扩散模型通过采样过程生成图像。前向过程在 T 步中向图像添加高斯噪声，直到它变成纯噪声。逆向过程学习去噪，逐步恢复原始图像。

```
前向过程（已知）：
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  其中 epsilon ~ N(0, I)

  经过 T 步：x_T ~ N(0, I)（纯噪声）

逆向过程（学习得到）：
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  其中 z ~ N(0, I)

  每一步去噪都是一步采样。
```

与本课方法的联系：
- 每一步去噪都使用重参数化技巧（采样噪声，应用确定性变换）
- 噪声调度 {alpha_t} 控制一种形式的温度退火
- 训练使用蒙特卡洛估计来近似 ELBO（证据下界）
- 扩散模型中的祖先采样是一条马尔可夫链（每一步只取决于当前状态）

整个图像生成过程就是迭代采样：从噪声开始，在每一步中，基于学习到的去噪模型，采样一个稍微不那么嘈杂的版本。

## 动手实现

### 步骤 1：均匀采样和逆 CDF 采样

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个指数分布样本并验证均值是否为 1/lambda。

### 步骤 2：拒绝采样

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

使用拒绝采样从截断正态分布中抽样。通过绘制样本的直方图来验证形状。

### 步骤 3：重要性采样

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

使用均匀提议估计正态分布下的 E[X^2]。与已知答案（mu^2 + sigma^2）进行比较。

### 步骤 4：蒙特卡洛估计圆周率

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 步骤 5：Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从双峰分布（两个高斯的混合）中采样。可视化链的轨迹。

### 步骤 6：Gibbs 采样

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 步骤 7：温度采样

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示温度如何改变一组词元 logit 的输出分布。

### 步骤 8：Top-k 和 top-p 采样

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 步骤 9：重参数化技巧

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

演示梯度可以通过重参数化的采样流动，但不能通过直接采样流动。

### 步骤 10：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示降低温度如何使输出趋近于 one-hot 向量。

完整实现及所有可视化代码在 `code/sampling.py` 中。

## 使用库实现

使用 NumPy 和 SciPy 的生产版本：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

大规模 MCMC 请使用专用库：
- PyMC：完整贝叶斯建模，支持 NUTS（自适应 HMC）
- emcee：集成 MCMC 采样器
- NumPyro/JAX：GPU 加速的 MCMC

你已经从零构建了这些方法。现在你知道库函数调用在做什么了。

## 练习

1. 为柯西分布实现逆 CDF 采样。CDF 为 F(x) = 0.5 + arctan(x)/pi。生成 10,000 个样本并将直方图与真实 PDF 对比。注意重尾特性（远离中心的极端值）。

2. 使用拒绝采样从 Beta(2, 5) 分布中生成样本，使用 Uniform(0, 1) 作为提议分布。将接受的样本与真实 Beta PDF 对比绘图。理论接受率是多少？

3. 使用 1,000、10,000 和 100,000 个样本的蒙特卡洛方法估计 sin(x) 从 0 到 pi 的积分。比较各级别的误差。验证误差是否按 O(1/sqrt(N)) 缩放。

4. 实现 Metropolis-Hastings 从二维分布 p(x, y) 正比于 exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2) 中采样。绘制样本和链的轨迹。实验不同的提议标准差。

5. 构建一个完整的文本生成演示：给定一个包含 10 个词及其 logit 的词汇表，使用 (a) 贪心、(b) 温度=0.7、(c) top-k=3、(d) top-p=0.9 生成 20 个词元的序列。比较 5 次运行的输出多样性。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| 采样（Sampling） | "抽取随机值" | 根据概率分布生成值。所有生成式 AI 背后的机制 |
| 均匀分布（Uniform Distribution） | "全部等概率" | [a, b] 中每个值的概率密度为 1/(b-a)。所有采样方法的起点 |
| 逆 CDF（Inverse CDF） | "概率变换" | F_inverse(U) 将均匀样本转换为具有已知 CDF 的任意分布的样本。精确且高效 |
| 拒绝采样（Rejection Sampling） | "提议并接受/拒绝" | 从简单的提议分布中生成，以正比于目标/提议比值的概率接受。精确但浪费样本 |
| 重要性采样（Importance Sampling） | "重加权样本" | 使用来自 q(x) 的样本，通过给每个样本加权 p(x)/q(x) 来估计 p(x) 下的期望。是强化学习 PPO 的核心 |
| 蒙特卡洛（Monte Carlo） | "随机样本取平均" | 用样本平均值近似积分。误差 O(1/sqrt(N))，与维度无关 |
| MCMC | "收敛的随机游走" | 构建一条马尔可夫链，其平稳分布是目标分布。Metropolis-Hastings 是基础算法 |
| Metropolis-Hastings | "接受上坡，有时接受下坡" | 提议移动，基于密度比决定是否接受。细致平衡确保收敛到目标分布 |
| Gibbs 采样 | "一次一个变量" | 固定其他变量，从条件分布更新每个变量。100% 接受率 |
| 温度（Temperature） | "置信度旋钮" | 在 softmax 之前将 logit 除以 T。T<1 使分布更尖锐（更确定），T>1 使分布更平坦（更多样） |
| Top-k 采样 | "保留前 k 个最佳" | 置零除概率最高的 k 个词元外的所有词元，重新归一化后采样。候选集大小固定 |
| 核采样（top-p） | "保留概率高的那些" | 保留累积概率超过 p 的最小词元集合。候选集大小自适应 |
| 重参数化技巧（Reparameterization Trick） | "将随机性移到外面" | 写成 z = mu + sigma * epsilon，其中 epsilon ~ N(0,1)。使采样可微分。VAE 训练的关键 |
| Gumbel-Softmax | "软类别采样" | 使用 Gumbel 噪声 + 带温度的 softmax 实现类别采样的可微近似 |
| 分层采样（Stratified Sampling） | "强制覆盖" | 将采样空间划分为层，从每层中采样。方差始终低于或等于朴素蒙特卡洛 |
| 预热期（Burn-in） | "热身阶段" | 在链达到平稳分布之前丢弃的初始 MCMC 样本 |
| 细致平衡（Detailed Balance） | "可逆性条件" | p(x) * T(x->y) = p(y) * T(y->x)。p 为马尔可夫链平稳分布的充分条件 |
| 扩散采样（Diffusion Sampling） | "迭代去噪" | 从噪声开始，通过学习到的去噪步骤生成数据。每一步都是条件采样操作 |

## 延伸阅读

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC 基础的详细教程
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Gumbel-Softmax 原始论文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - 核采样（top-p）论文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 引入重参数化技巧的 VAE 论文
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM 将采样与图像生成联系起来
