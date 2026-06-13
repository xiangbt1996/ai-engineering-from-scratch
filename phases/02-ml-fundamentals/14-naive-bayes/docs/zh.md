# 朴素贝叶斯

> "朴素"假设是错的，但它居然管用。这就是它的美妙之处。

**类型：** 动手实现
**语言：** Python
**前置课程：** 第二阶段，第01-07课（分类、贝叶斯定理）
**时长：** 约75分钟

## 学习目标

- 从零实现带拉普拉斯平滑（Laplace Smoothing）的多项式朴素贝叶斯（Multinomial Naive Bayes），用于文本分类
- 解释为什么朴素独立性假设在数学上是错误的，但在实践中仍能产生正确的类别排名
- 比较多项式（Multinomial）、伯努利（Bernoulli）和高斯（Gaussian）朴素贝叶斯变体，并为给定的特征类型选择正确的变体
- 在高维稀疏数据上将朴素贝叶斯与逻辑回归（Logistic Regression）进行比较，并解释其中的偏差-方差权衡

## 问题引入

你需要对文本进行分类。将邮件分为垃圾邮件和非垃圾邮件。将客户评论分为正面和负面。将工单分到不同类别。你有成千上万的特征（每个词一个）和有限的训练数据。

大多数分类器在这里都会崩溃。逻辑回归需要足够的样本来可靠地估计数千个权重。决策树每次只在一个词上分裂，严重过拟合。KNN 在10,000维空间中毫无意义，因为每个点到其他所有点的距离都差不多。

朴素贝叶斯（Naive Bayes）能处理这种情况。它做了一个数学上错误的假设（给定类别，每个特征与其他所有特征独立），但在文本分类上仍然能超越"更聪明"的模型，尤其是在小训练集上。它只需一次遍历数据就完成训练。它可以扩展到数百万个特征。它能产生概率估计（虽然由于独立性假设，这些概率通常校准不佳）。

理解为什么一个错误的假设却能产生好的预测，这教会你机器学习中一个根本性的道理：最好的模型不是最正确的那个，而是在你的数据上拥有最佳偏差-方差权衡（Bias-Variance Tradeoff）的那个。

## 核心概念

### 贝叶斯定理（快速回顾）

贝叶斯定理（Bayes' Theorem）翻转条件概率：

```
P(class | features) = P(features | class) * P(class) / P(features)
```

我们想要 `P(class | features)` -- 给定文档中的词，该文档属于某类的概率。我们可以从以下项计算：
- `P(features | class)` -- 似然（Likelihood），在该类别的文档中看到这些词的概率
- `P(class)` -- 先验（Prior）概率，该类别有多常见（垃圾邮件总体占多大比例？）
- `P(features)` -- 证据（Evidence），对所有类别相同，因此在比较时可以忽略

具有最高 `P(class | features)` 的类别胜出。

### 朴素独立性假设

精确计算 `P(features | class)` 需要估计所有特征的联合概率。词汇表有10,000个词时，你需要估计 2^10,000 种可能组合的分布，这是不可能的。

朴素假设（Naive Assumption）：给定类别，每个特征都是条件独立（Conditional Independence）的。

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

用 n 个简单的单特征分布代替一个不可能的联合分布。每个分布只需要一个计数。

这个假设显然是错误的。"machine"和"learning"这两个词在任何文档中都不是独立的。但分类器不需要正确的概率估计，它需要的是正确的排名——哪个类别的概率最高。独立性假设引入了系统性误差，但这些误差对所有类别的影响类似，因此排名保持正确。

### 为什么它仍然有效

三个原因：

1. **排名优先于校准。** 分类只需要排名最高的类别是正确的。即使 P(spam) = 0.99999 而真实概率是 0.7，分类器仍然会正确选择 spam。我们不需要正确的概率，我们需要正确的赢家。

2. **高偏差，低方差。** 独立性假设是一个强先验（Strong Prior）。它严格约束了模型，从而防止过拟合。在训练数据有限时，一个略有偏差但稳定的模型胜过一个理论上正确但极不稳定的模型。这正是偏差-方差权衡的体现。

3. **特征冗余相互抵消。** 相关特征提供冗余的证据。分类器对这些证据做了重复计算，但它对正确类别也做了同样的重复计算。如果"machine"和"learning"总是一起出现，两者都为"tech"类提供证据。朴素贝叶斯计算了两次，但它对正确的类别也计算了两次。

第四个实用原因：朴素贝叶斯极其快速。训练只需一次遍历数据来计算频率。预测是一次矩阵乘法。你可以在几秒钟内对一百万份文档进行训练。这种速度意味着你可以更快地迭代、尝试更多特征集、运行更多实验。

### 数学逐步推导

让我们通过一个具体例子来推导。假设我们有两个类别：垃圾邮件（spam）和非垃圾邮件（not-spam）。词汇表有三个词："free"、"money"、"meeting"。

训练数据：
- 垃圾邮件中提到"free"80次、"money"60次、"meeting"10次（共150个词）
- 非垃圾邮件中提到"free"5次、"money"10次、"meeting"100次（共115个词）
- 40%的邮件是垃圾邮件，60%是非垃圾邮件

使用拉普拉斯平滑（alpha=1）：

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

新邮件包含："free"（2次）、"money"（1次）、"meeting"（0次）。

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

垃圾邮件以较大的优势获胜。"free"出现两次是垃圾邮件的强烈证据。注意"meeting"未出现对两个对数和的贡献都是零（0 * log(P)）——在多项式朴素贝叶斯中，缺失的词没有影响。显式建模词的缺失是伯努利朴素贝叶斯的特征。

### 三种变体

朴素贝叶斯有三种变体。每种以不同方式建模 `P(feature | class)`。

#### 多项式朴素贝叶斯（Multinomial Naive Bayes）

将每个特征建模为计数。最适合以词频或 TF-IDF 值为特征的文本数据。

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

`alpha` 是拉普拉斯平滑（下文解释）。这个变体是文本分类的主力。

#### 高斯朴素贝叶斯（Gaussian Naive Bayes）

将每个特征建模为正态分布。最适合连续特征。

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

每个类别的每个特征都有自己的均值和方差。当特征在每个类别内确实近似服从钟形曲线时，效果很好。

#### 伯努利朴素贝叶斯（Bernoulli Naive Bayes）

将每个特征建模为二元值（出现或不出现）。最适合短文本或二元特征向量。

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

与多项式不同，伯努利会显式惩罚某个词的缺失。如果"free"通常出现在垃圾邮件中但在这封邮件里没有出现，伯努利会将此算作不是垃圾邮件的证据。

### 何时使用哪种变体

| 变体 | 特征类型 | 最适合 | 示例 |
|---------|-------------|----------|---------|
| 多项式（Multinomial） | 计数或频率 | 文本分类、词袋模型 | 邮件垃圾过滤、主题分类 |
| 高斯（Gaussian） | 连续值 | 特征近似正态分布的表格数据 | 鸢尾花分类、传感器数据 |
| 伯努利（Bernoulli） | 二元值（0/1） | 短文本、二元特征向量 | 短信垃圾过滤、存在/缺失特征 |

### 拉普拉斯平滑

当一个词出现在测试数据中，但在训练数据的某个类别中从未出现过，会发生什么？

不做平滑时：`P(word | class) = 0/N = 0`。一个零值乘以整个乘积会使 `P(class | features) = 0`，无论其他证据有多强。一个未见过的词就会摧毁整个预测，无论有多少其他证据支持它。

拉普拉斯平滑（Laplace Smoothing）给每个特征计数加上一个小的计数 `alpha`（通常为1）：

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

当 alpha=1 时，每个词至少有一个微小的概率。测试邮件中出现的词"discombobulate"不再会杀死垃圾邮件的概率。平滑有贝叶斯解释：它等价于在词分布上放置一个均匀的狄利克雷先验（Dirichlet Prior）。

更高的 alpha 意味着更强的平滑（更均匀的分布）。更低的 alpha 意味着模型更信任数据。Alpha 是一个需要调优的超参数。

alpha 的效果：

| Alpha | 效果 | 何时使用 |
|-------|--------|-------------|
| 0.001 | 几乎不平滑，信任数据 | 非常大的训练集，不预期出现未见特征 |
| 0.1 | 轻微平滑 | 大训练集 |
| 1.0 | 标准拉普拉斯平滑 | 默认起始点 |
| 10.0 | 重度平滑，压平分布 | 非常小的训练集，预期出现很多未见特征 |

### 对数空间计算

将数百个概率（每个都小于1）相乘会导致浮点数下溢。即使真实值是一个很小的正数，乘积在浮点运算中也会变成零。

解决方案：在对数空间中工作。不乘概率，而是加它们的对数：

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

这将预测转化为一个点积：

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

矩阵乘法。这就是朴素贝叶斯预测如此快速的原因——它和单层线性模型是同一种运算。

### 朴素贝叶斯 vs 逻辑回归

两者都是文本的线性分类器。区别在于它们建模的对象不同。

| 方面 | 朴素贝叶斯 | 逻辑回归 |
|--------|------------|-------------------|
| 类型 | 生成式（建模 P(X\|Y)） | 判别式（建模 P(Y\|X)） |
| 训练 | 统计频率 | 优化损失函数 |
| 小数据 | 更好（强先验有帮助） | 更差（不够估计权重） |
| 大数据 | 更差（错误假设会拖后腿） | 更好（灵活的决策边界） |
| 特征 | 假设独立 | 能处理相关性 |
| 速度 | 单次遍历，非常快 | 迭代优化 |
| 校准 | 概率估计差 | 概率估计好 |

经验法则：先用朴素贝叶斯。如果你有足够的数据且 NB 到达瓶颈，再切换到逻辑回归。

### 分类流水线

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

在实践中，我们在对数空间中工作以避免浮点数下溢。不是乘以许多小概率，而是加上它们的对数：

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

## 动手实现

`code/naive_bayes.py` 中的代码从零实现了多项式朴素贝叶斯和高斯朴素贝叶斯。

### MultinomialNB

从零实现：

1. **fit(X, y)**：对于每个类别，统计每个特征的频率。添加拉普拉斯平滑。计算对数概率。存储类别先验（类别频率的对数）。

2. **predict_log_proba(X)**：对于每个样本，计算所有类别的 log P(class) + sum of log P(feature_i | class)。这是一个矩阵乘法：X @ log_probs.T + log_priors。

3. **predict(X)**：返回对数概率最高的类别。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键洞察：拟合后，预测只是矩阵乘法加偏置。这就是朴素贝叶斯如此快速的原因。

### GaussianNB

对于连续特征，我们估计每个类别每个特征的均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测使用每个特征的高斯概率密度函数，跨特征相乘（在对数空间中相加）。

### 演示：文本分类

代码生成模拟两个类别（技术文章 vs 体育文章）的合成词袋数据。每个类别有不同的词频分布。MultinomialNB 使用词计数进行分类。

合成数据的工作方式如下：我们创建200个"词"（特征列）。词 0-39 在技术文章中出现频率高，在体育中低。词 80-119 在体育中高，在技术中低。词 40-79 在两者中都是中等频率。这创造了一个真实的场景，其中一些词是强类别指示器，其他的是噪声。

### 演示：连续特征

代码生成类似鸢尾花的数据（3个类别、4个特征、高斯聚类）。GaussianNB 使用每个类别的均值和方差进行分类。每个类别有不同的中心（均值向量）和不同的分散程度（方差），模拟真实世界中不同类别之间测量值系统性差异的数据。

代码还演示了：
- **平滑比较：** 使用不同的 alpha 值训练 MultinomialNB，展示平滑强度对准确率的影响。
- **训练规模实验：** 当训练数据从20增长到1600个样本时，NB 准确率如何提升。NB 即使在非常少的样本下也能达到不错的准确率——这是它的主要优势。
- **混淆矩阵：** 每类的精确率、召回率和 F1 分数，展示 NB 在哪里犯错。

### 预测速度

朴素贝叶斯的预测是一次矩阵乘法。对于 n 个样本、d 个特征和 k 个类别：
- MultinomialNB：一次矩阵乘法 (n x d) @ (d x k) = O(n * d * k)
- GaussianNB：n * k 次高斯 PDF 计算，每次对 d 个特征 = O(n * d * k)

两者在每个维度上都是线性的。与 KNN（需要计算到所有训练点的距离）或带 RBF 核的 SVM（需要对所有支持向量进行核计算）相比，NB 在预测时快了几个数量级。

## 实际使用

使用 sklearn，两种变体都只需一行代码：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

使用 sklearn 进行文本分类：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

`naive_bayes.py` 中的代码将从零实现与 sklearn 在相同数据上进行比较，以验证正确性。

### TF-IDF 与朴素贝叶斯

原始词计数对每个词的每次出现赋予相同的权重。但像"the"和"is"这样的常见词在每个类别中都频繁出现——它们不携带任何信息。TF-IDF（词频-逆文档频率）降低常见词的权重，提高罕见的、有区分力的词的权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF 值是非负的，因此可以与 MultinomialNB 配合使用。TF-IDF + MultinomialNB 的组合是文本分类最强的基线之一。在训练样本少于10,000个的数据集上，它经常击败更复杂的模型。

### BernoulliNB 用于短文本

对于短文本（推文、短信、聊天消息），BernoulliNB 可能优于 MultinomialNB。短文本的词计数较低，MultinomialNB 依赖的频率信息噪声很大。BernoulliNB 只关注存在与否，这在短文本中更可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

CountVectorizer 中的 `binary=True` 参数将所有计数转换为 0/1。不设置时，BernoulliNB 仍然可以工作，但它看到的是它本来不是为之设计的计数值。

### 校准 NB 概率

NB 的概率校准很差。当 NB 说 P(spam) = 0.95 时，真实概率可能是 0.7。如果你需要可靠的概率估计（例如设置阈值或与其他模型组合），可以使用 sklearn 的 CalibratedClassifierCV：

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

这在 NB 的原始分数上通过交叉验证拟合了一个逻辑回归。得到的概率更接近真实的类别频率。

### 常见陷阱

1. **负特征值。** MultinomialNB 要求特征值非负。如果你有负值（如某些设置下的 TF-IDF 或标准化后的特征），请改用 GaussianNB，或将特征平移为正值。

2. **零方差特征。** GaussianNB 除以方差。如果某个特征在某个类别中方差为零（所有值相同），概率计算会出错。代码中对所有方差添加了一个小的平滑项（1e-9）来防止这种情况。

3. **类别不平衡。** 如果99%的邮件是非垃圾邮件，先验 P(not-spam) = 0.99 太强了，会压制似然证据。你可以手动设置类别先验，或在 sklearn 中使用 class_prior 参数。

4. **特征缩放。** MultinomialNB 不需要缩放（它处理计数）。GaussianNB 也不需要缩放（它估计每个特征的统计量）。这是相对于逻辑回归和 SVM 的优势，后者对特征尺度敏感。

## 交付物

本课产出：
- `outputs/skill-naive-bayes-chooser.md` -- 用于选择正确 NB 变体的决策技能
- `code/naive_bayes.py` -- 从零实现的 MultinomialNB 和 GaussianNB，附 sklearn 对比

### 朴素贝叶斯何时失败

当独立性假设导致错误的排名（不仅仅是错误的概率）时，NB 就会失败。这发生在以下情况：

1. **强特征交互。** 如果类别取决于两个特征的组合而非单独任一个（类似异或的模式），NB 将完全忽略它。单个特征本身不提供证据，而 NB 无法非线性地组合它们。

2. **高度相关的特征具有相反的证据。** 如果特征 A 说"垃圾邮件"而特征 B 说"非垃圾邮件"，但 A 和 B 完全相关（实际上它们总是一致的），NB 会在不存在冲突的地方看到冲突的证据。

3. **非常大的训练集。** 有了足够的数据，判别式模型（如逻辑回归）能学到真实的决策边界并超越 NB。在小数据上有帮助的独立性假设现在反而成了模型的瓶颈。

在实践中，这些失败模式在文本分类中很少见。文本特征数量多、单个特征弱，独立性假设的误差倾向于相互抵消。对于具有少量强相关特征的表格数据，优先考虑逻辑回归或基于树的模型。

## 练习

1. **平滑实验。** 在文本数据上使用 alpha 值为 0.01、0.1、1.0、10.0 和 100.0 训练 MultinomialNB。绘制准确率与 alpha 的关系图。性能在哪里达到峰值？为什么非常高的 alpha 会降低性能？

2. **特征独立性检验。** 取一个真实的文本数据集。选两个明显相关的词（"machine"和"learning"）。计算 P(word1 | class) * P(word2 | class) 并与 P(word1 AND word2 | class) 比较。独立性假设有多错？这会影响分类准确率吗？

3. **伯努利实现。** 扩展代码，添加一个 BernoulliNB 类。将词袋转换为二元（存在/缺失）并将其与 MultinomialNB 在文本数据上的准确率进行比较。伯努利在何时胜出？

4. **NB vs 逻辑回归。** 在文本数据上训练两者。从100个训练样本开始，增加到10,000个。绘制两者的准确率与训练集大小的关系图。逻辑回归在什么时候超越朴素贝叶斯？

5. **垃圾邮件过滤器。** 构建一个完整的垃圾邮件分类器：对原始邮件文本分词、构建词汇表、创建词袋特征、训练 MultinomialNB、使用精确率和召回率评估（为什么不仅仅用准确率？）。

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 朴素贝叶斯（Naive Bayes） | "简单的概率分类器" | 应用贝叶斯定理并假设给定类别时特征条件独立的分类器 |
| 条件独立（Conditional Independence） | "特征互不影响" | P(A, B \| C) = P(A \| C) * P(B \| C) -- 在已知 C 的情况下，知道 B 不会对 A 提供新的信息 |
| 拉普拉斯平滑（Laplace Smoothing） | "加一平滑" | 给每个特征添加一个小计数，防止零概率支配整个预测 |
| 先验（Prior） | "看到数据之前你相信什么" | P(class) -- 在观察任何特征之前每个类别的概率 |
| 似然（Likelihood） | "数据有多契合" | P(features \| class) -- 如果类别已知，观察到这些特征的概率 |
| 后验（Posterior） | "看到数据之后你相信什么" | P(class \| features) -- 在观察到特征后，类别的更新概率 |
| 生成式模型（Generative Model） | "建模数据如何生成" | 学习 P(X \| Y) 和 P(Y)，然后用贝叶斯定理得到 P(Y \| X) 的模型 |
| 判别式模型（Discriminative Model） | "建模决策边界" | 直接学习 P(Y \| X) 而不建模 X 如何生成的模型 |
| 对数概率（Log Probability） | "避免下溢" | 使用 log P 而不是 P，防止许多小数的乘积在浮点运算中变为零 |

## 延伸阅读

- [scikit-learn 朴素贝叶斯文档](https://scikit-learn.org/stable/modules/naive_bayes.html) -- 三种变体的数学细节
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) -- 多项式与伯努利文本分类的经典比较
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) -- 对文本 NB 的改进
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) -- 证明 NB 在小数据下比 LR 收敛更快
