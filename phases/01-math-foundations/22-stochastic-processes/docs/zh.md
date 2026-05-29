# 随机过程

> 有结构的随机性。随机游走、马尔可夫链和扩散模型背后的数学。

**类型：** 学习理解
**语言：** Python
**前置知识：** 第一阶段，第06-07课（概率论、贝叶斯）
**时间：** 约75分钟

## 学习目标

- 模拟一维和二维随机游走（Random Walk），并验证位移的 sqrt(n) 缩放规律
- 构建马尔可夫链（Markov Chain）模拟器，通过特征分解计算其平稳分布
- 实现 Metropolis-Hastings MCMC 和朗之万动力学（Langevin Dynamics），用于从目标分布中采样
- 将前向扩散过程（Forward Diffusion）与布朗运动联系起来，并解释反向过程如何生成数据

## 问题背景

许多 AI 系统涉及随时间演化的随机性。不是静态的随机性——而是有结构的、序列化的随机性，其中每一步都依赖于前一步的结果。

语言模型逐个生成 token。每个 token 依赖于之前的上下文。模型输出一个概率分布，从中采样，然后继续。这就是一个随机过程。

扩散模型逐步向图像添加噪声，直到它变成纯粹的静态噪声。然后反转这个过程，逐步去噪直到新图像浮现出来。前向过程是一个马尔可夫链。反向过程是一个学习到的反向马尔可夫链。

强化学习智能体在环境中采取行动。每个行动以一定概率导向新的状态。智能体在随机的世界中遵循随机的策略。整个过程是一个马尔可夫决策过程（Markov Decision Process）。

MCMC 采样——贝叶斯推断的支柱——构造一个马尔可夫链，使其平稳分布恰好是你想要采样的后验分布。

所有这些都建立在四个基础概念之上：
1. 随机游走——最简单的随机过程
2. 马尔可夫链——具有转移矩阵的结构化随机性
3. 朗之万动力学——带噪声的梯度下降
4. Metropolis-Hastings——从任意分布中采样

## 核心概念

### 随机游走

从位置 0 出发。每一步抛一枚公平硬币。正面：向右移动 (+1)。反面：向左移动 (-1)。

经过 n 步后，你的位置是 n 个随机 +/-1 值的总和。期望位置为 0（游走是无偏的）。但离原点的期望距离以 sqrt(n) 的速率增长。

这是反直觉的。游走是公平的——没有向任何方向的漂移。但随着时间推移，它会越来越远离起点。n 步后的标准差是 sqrt(n)。

```
第0步:  位置 = 0
第1步:  位置 = +1 或 -1
第2步:  位置 = +2, 0, 或 -2
...
第100步: 离原点的期望距离 ~ 10 (sqrt(100))
第10000步: 离原点的期望距离 ~ 100 (sqrt(10000))
```

**在二维中**，游走以等概率向上、下、左、右移动。离原点距离同样满足 sqrt(n) 的缩放规律。路径呈现出类似分形的模式。

**为什么是 sqrt(n)？** 每一步是 +1 或 -1，概率相等。n 步后，位置 S_n = X_1 + X_2 + ... + X_n，其中每个 X_i 取 +/-1。每一步的方差为 1，且各步独立，所以 Var(S_n) = n。标准差 = sqrt(n)。根据中心极限定理，S_n / sqrt(n) 收敛到标准正态分布。

这个 sqrt(n) 的缩放规律在机器学习中随处可见。SGD 噪声以 1/sqrt(batch_size) 的比例缩放。嵌入维度以 sqrt(d) 缩放。平方根是独立随机叠加的标志。

**与布朗运动的联系。** 取步长为 1/sqrt(n)、每个单位时间走 n 步的随机游走。当 n 趋向无穷时，游走收敛到布朗运动 B(t)——一个连续时间过程，其中 B(t) 服从均值为 0、方差为 t 的正态分布。

布朗运动是扩散的数学基础。它描述了流体中粒子的随机抖动、股票价格的波动，以及——关键的——扩散模型中的噪声过程。

**赌徒破产问题。** 一个随机游走者从位置 k 出发，在 0 和 N 处有吸收壁。到达 N 之前不被 0 吸收的概率是多少？对于公平游走：P(到达 N) = k/N。这个结果出人意料地简洁优美。它与鞅（Martingale）理论相关——公平随机游走是一个鞅（未来的期望值等于当前值）。

### 马尔可夫链

马尔可夫链是一个按照固定概率在状态之间转移的系统。关键性质是：下一个状态只依赖于当前状态，而不依赖于历史。

```
P(X_{t+1} = j | X_t = i, X_{t-1} = ...) = P(X_{t+1} = j | X_t = i)
```

这就是马尔可夫性质。这意味着你可以用一个转移矩阵 P 来描述整个动力学：

```
P[i][j] = 从状态 i 转移到状态 j 的概率
```

P 的每一行之和为 1（你必须转移到某个状态）。

**示例——天气：**

```
状态: 晴天 (0), 雨天 (1), 多云 (2)

P = [[0.7, 0.1, 0.2],    （如果是晴天：70% 晴天，10% 雨天，20% 多云）
     [0.3, 0.4, 0.3],    （如果是雨天：30% 晴天，40% 雨天，30% 多云）
     [0.4, 0.2, 0.4]]    （如果是多云：40% 晴天，20% 雨天，40% 多云）
```

从任意状态出发。经过多次转移后，状态的分布收敛到平稳分布 pi，满足 pi * P = pi。这是 P 的特征值为 1 的左特征向量。

对于天气链，平稳分布可能是 [0.53, 0.18, 0.29]——从长期来看，无论初始状态是什么，53% 的时间是晴天。

```mermaid
graph LR
    S["晴天"] -->|0.7| S
    S -->|0.1| R["雨天"]
    S -->|0.2| C["多云"]
    R -->|0.3| S
    R -->|0.4| R
    R -->|0.3| C
    C -->|0.4| S
    C -->|0.2| R
    C -->|0.4| C
```

**计算平稳分布。** 有两种方法：

1. **幂方法**：将任意初始分布反复乘以 P。经过足够多次迭代后收敛。
2. **特征值方法**：找到 P 的特征值为 1 的左特征向量。即 P^T 的特征值为 1 的特征向量。

两种方法都要求链满足收敛条件。

**收敛条件。** 一个马尔可夫链在以下条件下收敛到唯一的平稳分布：
- **不可约**：每个状态都可以从任何其他状态到达
- **非周期**：链不会以固定周期循环

你在机器学习中遇到的大多数链都满足这两个条件。

**吸收状态。** 如果一旦进入某个状态就永远不会离开（P[i][i] = 1），该状态就是吸收状态。吸收马尔可夫链用于建模有终止状态的过程——一个结束的游戏、一个流失的客户、一个遇到文本结束标记的 token 序列。

**混合时间（Mixing Time）。** 需要多少步链才"接近"平稳分布？形式化地说，是总变差距离降到某个阈值以下所需的步数。快速混合 = 所需步数少。P 的谱间隙（1 减去第二大特征值）控制混合时间。间隙越大 = 混合越快。

### 与语言模型的联系

语言模型中的 token 生成近似于一个马尔可夫过程。给定当前上下文，模型输出下一个 token 的分布。温度控制分布的尖锐程度：

```
P(token_i) = exp(logit_i / temperature) / sum(exp(logit_j / temperature))
```

- Temperature = 1.0：标准分布
- Temperature < 1.0：更尖锐（更确定性）
- Temperature > 1.0：更平坦（更随机）
- Temperature -> 0：取最大值（贪心策略）

Top-k 采样截断到概率最高的 k 个 token。Top-p（核采样）截断到累积概率超过 p 的最小 token 集合。两者都修改了马尔可夫转移概率。

### 布朗运动

随机游走的连续时间极限。位置 B(t) 有三个性质：
1. B(0) = 0
2. B(t) - B(s) 服从均值为 0、方差为 t - s 的正态分布（t > s 时）
3. 不重叠时间区间上的增量是独立的

布朗运动是连续但处处不可微的——它在每个尺度上都在抖动。其路径在平面上的分形维数为 2。

在离散模拟中，你通过以下方式近似布朗运动：

```
B(t + dt) = B(t) + sqrt(dt) * z,    其中 z ~ N(0, 1)
```

sqrt(dt) 的缩放很重要。它来自对随机游走应用中心极限定理。

### 朗之万动力学

梯度下降找到函数的最小值。朗之万动力学找到正比于 exp(-U(x)/T) 的概率分布，其中 U 是能量函数，T 是温度。

```
x_{t+1} = x_t - dt * gradient(U(x_t)) + sqrt(2 * T * dt) * z_t
```

两种力作用于粒子：
1. **梯度力** (-dt * gradient(U))：推向低能量区域（类似梯度下降）
2. **随机力** (sqrt(2*T*dt) * z)：推向随机方向（探索）

当温度 T = 0 时，这就是纯粹的梯度下降。在高温下，它几乎是随机游走。在合适的温度下，粒子探索能量景观，在低能量区域停留更多时间。

**与扩散模型的联系。** 扩散模型的前向过程是：

```
x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * noise
```

这是一个逐渐将数据与噪声混合的马尔可夫链。经过足够多步后，x_T 变为纯高斯噪声。

反向过程——从噪声回到数据——也是一个马尔可夫链，但其转移概率由神经网络学习。网络学习预测每一步添加的噪声，然后将其减去。

```mermaid
graph LR
    subgraph "前向过程（添加噪声）"
        X0["x_0（数据）"] -->|"+ 噪声"| X1["x_1"]
        X1 -->|"+ 噪声"| X2["x_2"]
        X2 -->|"..."| XT["x_T（纯噪声）"]
    end
    subgraph "反向过程（去噪）"
        XT2["x_T（噪声）"] -->|"神经网络"| XR2["x_{T-1}"]
        XR2 -->|"神经网络"| XR1["x_{T-2}"]
        XR1 -->|"..."| XR0["x_0（生成的数据）"]
    end
```

### MCMC：马尔可夫链蒙特卡洛

有时你需要从一个概率分布 p(x) 中采样，你能计算它的值（精确到一个常数倍），但无法直接从中采样。贝叶斯后验分布就是经典的例子——你知道似然乘以先验，但归一化常数无法计算。

**Metropolis-Hastings** 算法构造一个马尔可夫链，使其平稳分布为 p(x)：

1. 从某个位置 x 开始
2. 从提议分布 Q(x'|x) 中提议一个新位置 x'
3. 计算接受比率：a = p(x') * Q(x|x') / (p(x) * Q(x'|x))
4. 以概率 min(1, a) 接受 x'。否则留在 x。
5. 重复。

如果 Q 是对称的（例如 Q(x'|x) = Q(x|x') = N(x, sigma^2)），比率简化为 a = p(x') / p(x)。你只需要概率之比——归一化常数被消掉了。

在温和的条件下，该链保证收敛到 p(x)。但如果提议步长太小（随机游走）或太大（高拒绝率），收敛可能很慢。调节提议分布是 MCMC 的艺术。

**为什么它有效。** 接受比率确保了细致平衡（Detailed Balance）：处于 x 并移动到 x' 的概率等于处于 x' 并移动到 x 的概率。细致平衡意味着 p(x) 是链的平稳分布。因此经过足够多步后，样本来自 p(x)。

**实践注意事项：**
- **预热期（Burn-in）**：丢弃前 N 个样本。链需要时间从初始点到达平稳分布。
- **稀释（Thinning）**：每隔 k 个保留一个样本，以减少自相关。
- **多链**：从不同起点运行多条链。如果它们收敛到相同的分布，就是收敛的证据。
- **接受率**：对于 d 维的高斯提议，最优接受率约为 23%（Roberts & Rosenthal, 2001）。太高意味着链几乎不移动。太低意味着它拒绝一切。

### AI 中的随机过程

| 过程 | AI 应用 |
|------|---------|
| 随机游走 | 强化学习中的探索、Node2Vec 嵌入 |
| 马尔可夫链 | 文本生成、MCMC 采样 |
| 布朗运动 | 扩散模型（前向过程） |
| 朗之万动力学 | 基于分数的生成模型、SGLD |
| 马尔可夫决策过程 | 强化学习 |
| Metropolis-Hastings | 贝叶斯推断、后验采样 |

## 动手构建

### 第1步：随机游走模拟器

```python
import numpy as np

def random_walk_1d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    steps = rng.choice([-1, 1], size=n_steps)
    positions = np.concatenate([[0], np.cumsum(steps)])
    return positions


def random_walk_2d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    directions = rng.choice(4, size=n_steps)
    dx = np.zeros(n_steps)
    dy = np.zeros(n_steps)
    dx[directions == 0] = 1   # right
    dx[directions == 1] = -1  # left
    dy[directions == 2] = 1   # up
    dy[directions == 3] = -1  # down
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])
    return x, y
```

一维游走存储累积和。每一步是 +1 或 -1。n 步后的位置就是总和。方差随 n 线性增长，因此标准差以 sqrt(n) 增长。

### 第2步：马尔可夫链

```python
class MarkovChain:
    def __init__(self, transition_matrix, state_names=None):
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = len(self.P)
        self.state_names = state_names or [str(i) for i in range(self.n_states)]

    def step(self, current_state, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        probs = self.P[current_state]
        return rng.choice(self.n_states, p=probs)

    def simulate(self, start_state, n_steps, seed=None):
        rng = np.random.RandomState(seed)
        states = [start_state]
        current = start_state
        for _ in range(n_steps):
            current = self.step(current, rng)
            states.append(current)
        return states

    def stationary_distribution(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        return np.abs(stationary)
```

平稳分布是 P 的特征值为 1 的左特征向量。我们通过计算 P^T 的特征向量来找到它（转置将左特征向量变成右特征向量）。

### 第3步：朗之万动力学

```python
def langevin_dynamics(grad_U, x0, dt, temperature, n_steps, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        noise = rng.randn(*x.shape)
        x = x - dt * grad_U(x) + np.sqrt(2 * temperature * dt) * noise
        trajectory.append(x.copy())
    return np.array(trajectory)
```

梯度将 x 推向低能量区域。噪声防止它陷入局部最小值。在平衡态时，样本的分布正比于 exp(-U(x)/temperature)。

### 第4步：Metropolis-Hastings

```python
def metropolis_hastings(target_log_prob, proposal_std, x0, n_samples, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    samples = [x.copy()]
    accepted = 0
    for _ in range(n_samples - 1):
        x_proposed = x + rng.randn(*x.shape) * proposal_std
        log_ratio = target_log_prob(x_proposed) - target_log_prob(x)
        if np.log(rng.rand()) < log_ratio:
            x = x_proposed
            accepted += 1
        samples.append(x.copy())
    acceptance_rate = accepted / (n_samples - 1)
    return np.array(samples), acceptance_rate
```

算法提议一个新的点，检查它是否有更高的概率（或以概率比率的比例接受），然后重复。接受率应该在 23-50% 之间以实现良好的混合。

## 实际使用

在实践中，你会使用成熟的库来实现这些算法。但理解其机制对调试和调参很重要。

```python
import numpy as np

rng = np.random.RandomState(42)
walk = np.cumsum(rng.choice([-1, 1], size=10000))
print(f"Final position: {walk[-1]}")
print(f"Expected distance: {np.sqrt(10000):.1f}")
print(f"Actual distance: {abs(walk[-1])}")
```

### NumPy 处理转移矩阵

```python
import numpy as np

P = np.array([[0.7, 0.1, 0.2],
              [0.3, 0.4, 0.3],
              [0.4, 0.2, 0.4]])

distribution = np.array([1.0, 0.0, 0.0])
for _ in range(100):
    distribution = distribution @ P

print(f"Stationary distribution: {np.round(distribution, 4)}")
```

将初始分布反复乘以 P。经过足够多次迭代后，无论你从哪里开始，它都会收敛到平稳分布。这就是寻找主要左特征向量的幂方法。

### 与实际框架的联系

- **PyTorch 扩散：** Hugging Face `diffusers` 中的 `DDPMScheduler` 实现了前向和反向马尔可夫链
- **NumPyro / PyMC：** 使用 MCMC（NUTS 采样器，改进了 Metropolis-Hastings）进行贝叶斯推断
- **Gymnasium（强化学习）：** 环境的 step 函数定义了马尔可夫决策过程

### 验证马尔可夫链收敛

```python
import numpy as np

P = np.array([[0.9, 0.1], [0.3, 0.7]])

eigenvalues = np.linalg.eigvals(P)
spectral_gap = 1 - sorted(np.abs(eigenvalues))[-2]
print(f"Eigenvalues: {eigenvalues}")
print(f"Spectral gap: {spectral_gap:.4f}")
print(f"Approximate mixing time: {1/spectral_gap:.1f} steps")
```

谱间隙告诉你链遗忘初始状态的速度。间隙为 0.2 意味着大约需要 5 步混合。间隙为 0.01 意味着大约需要 100 步。在运行长时间模拟之前一定要检查这个值——混合缓慢的链会浪费计算资源。

## 产出物

本课产出：
- `outputs/prompt-stochastic-process-advisor.md`——一个帮助识别给定问题适用哪种随机过程框架的提示词

## 关联知识

| 概念 | 出现场景 |
|------|---------|
| 随机游走 | Node2Vec 图嵌入、强化学习中的探索 |
| 马尔可夫链 | 大语言模型中的 token 生成、MCMC 采样 |
| 布朗运动 | DDPM 的前向扩散过程、基于 SDE 的模型 |
| 朗之万动力学 | 基于分数的生成模型、随机梯度朗之万动力学（SGLD） |
| 平稳分布 | MCMC 收敛目标、PageRank |
| Metropolis-Hastings | 贝叶斯后验采样、模拟退火 |
| 温度 | LLM 采样、强化学习中的 Boltzmann 探索、模拟退火 |
| 混合时间 | MCMC 收敛速度、谱间隙分析 |
| 吸收状态 | 序列结束 token、强化学习中的终止状态 |
| 细致平衡 | MCMC 采样器的正确性保证 |

扩散模型值得特别关注。DDPM（Ho et al., 2020）定义了一个前向马尔可夫链：

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * I)
```

其中 beta_t 是噪声调度。经过 T 步后，x_T 近似于 N(0, I)。反向过程由神经网络参数化，预测噪声：

```
p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
```

生成的每一步都是在学习到的马尔可夫链中走一步。理解马尔可夫链就是理解扩散模型如何以及为何能生成数据。

SGLD（随机梯度朗之万动力学）将小批量梯度下降与朗之万噪声结合。不计算完整梯度，而是使用随机估计并添加校准过的噪声。随着学习率衰减，SGLD 从优化过渡到采样——你几乎免费获得了近似的贝叶斯后验样本。这是从神经网络获取不确定性估计的最简单方法之一。

所有这些联系背后的核心洞见是：随机过程不仅仅是理论工具。它们是现代 AI 系统内部的计算机制。当你调节 LLM 的温度时，你在调整一个马尔可夫链。当你训练扩散模型时，你在学习反转一个类似布朗运动的过程。当你运行贝叶斯推断时，你在构造一个收敛到后验分布的链。

## 练习

1. **模拟1000条10000步的随机游走。** 绘制最终位置的分布。验证它近似于均值为0、标准差为 sqrt(10000) = 100 的高斯分布。

2. **使用马尔可夫链构建文本生成器。** 在一个小型语料库上训练：对每个词统计到下一个词的转移。构建转移矩阵。通过从链中采样生成新句子。

3. **使用 Metropolis-Hastings 实现模拟退火。** 从高温开始（几乎接受一切）逐渐冷却（只接受改进）。用它来找到有多个局部最小值的函数的最小值。

4. **比较不同温度下的朗之万动力学。** 从双势阱势能 U(x) = (x^2 - 1)^2 中采样。在低温下，样本聚集在一个势阱中。在高温下，它们分布在两个势阱中。找到链在两个势阱之间混合的临界温度。

5. **实现前向扩散过程。** 从一个一维信号开始（例如正弦波）。使用线性噪声调度在100步内逐步添加噪声。展示信号如何退化为纯噪声。然后实现一个简单的去噪器来反转该过程（即使是一个简单地减去估计噪声的朴素版本也行）。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|---------|---------|
| 随机游走 | "抛硬币移动" | 位置在每一步按随机增量变化的过程 |
| 马尔可夫性质 | "无记忆性" | 未来只取决于当前状态，不取决于历史 |
| 转移矩阵 | "概率表" | P[i][j] = 从状态 i 移动到状态 j 的概率 |
| 平稳分布 | "长期平均" | 满足 pi*P = pi 的分布——链的平衡态 |
| 布朗运动 | "随机抖动" | 随机游走的连续时间极限，B(t) ~ N(0, t) |
| 朗之万动力学 | "带噪声的梯度下降" | 结合确定性梯度和随机扰动的更新规则 |
| MCMC | "走向目标" | 构造一个平稳分布恰好是目标分布的马尔可夫链 |
| Metropolis-Hastings | "提议并接受/拒绝" | 使用接受比率确保收敛的 MCMC 算法 |
| 温度 | "随机性旋钮" | 控制探索与利用之间权衡的参数 |
| 扩散过程 | "噪声进，噪声出" | 前向：逐步添加噪声。反向：逐步去除噪声。生成数据。 |

## 延伸阅读

- **Ho, Jain, Abbeel (2020)** —— "Denoising Diffusion Probabilistic Models"。引发扩散模型革命的 DDPM 论文。清晰地推导了前向和反向马尔可夫链。
- **Song & Ermon (2019)** —— "Generative Modeling by Estimating Gradients of the Data Distribution"。使用朗之万动力学进行采样的基于分数的方法。
- **Roberts & Rosenthal (2004)** —— "General state space Markov chains and MCMC algorithms"。MCMC 何时以及为何有效的理论。
- **Norris (1997)** —— "Markov Chains"。标准教材。涵盖收敛、平稳分布和首达时间。
- **Welling & Teh (2011)** —— "Bayesian Learning via Stochastic Gradient Langevin Dynamics"。将 SGD 与朗之万动力学结合，实现可扩展的贝叶斯推断。
