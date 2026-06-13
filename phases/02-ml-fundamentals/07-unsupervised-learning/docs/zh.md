# 无监督学习

> 没有标签，没有老师。算法自己发现结构。

**类型：** 动手实现
**语言：** Python
**前置知识：** 第 1 阶段（范数与距离、概率与分布）、第 2 阶段第 1-6 课
**时长：** 约 90 分钟

## 学习目标

- 从零实现 K 均值（K-Means）、DBSCAN 和高斯混合模型（GMM），并比较它们的聚类行为
- 使用轮廓系数（Silhouette Score）和肘部法则（Elbow Method）评估聚类质量，以选择最优的 K
- 解释 DBSCAN 何时优于 K 均值，并识别哪种算法能处理非球形聚类和异常值
- 使用聚类方法构建异常检测（Anomaly Detection）流水线，标记偏离正常模式的数据点

## 问题引入

到目前为止，每节机器学习课程都假设有标注数据："这是输入，这是正确的输出。"在现实世界中，标签是昂贵的。一家医院有数百万条患者记录，但没有人手动将每条记录标注疾病类别。一个电商网站有数百万次用户会话，但没有人手动标注客户分群。一个安全团队有网络日志，但没有人标记每个异常。

无监督学习（Unsupervised Learning）在没有被告知"要找什么"的情况下发现模式。它将相似的数据点分组、发现隐藏结构，并识别异常。如果监督学习是对照答案书学习，那么无监督学习就是盯着原始数据直到模式自己浮现。

问题在于：没有标签，你无法直接衡量"对"或"错"。你需要不同的工具来评估算法发现的结构是否有意义。

## 核心概念

### 聚类：将相似的事物分组

聚类（Clustering）将每个数据点分配到一个组（簇），使得同组内的点彼此比与其他组的点更相似。问题始终是："相似"是什么意思？

```mermaid
flowchart LR
    A[Raw Data] --> B{Choose Method}
    B --> C[K-Means]
    B --> D[DBSCAN]
    B --> E[Hierarchical]
    B --> F[GMM]
    C --> G[Flat, spherical clusters]
    D --> H[Arbitrary shapes, noise detection]
    E --> I[Tree of nested clusters]
    F --> J[Soft assignments, elliptical clusters]
```

### K 均值：主力算法

K 均值将数据精确地划分为 K 个簇。每个簇有一个质心（Centroid，即质量中心），每个点属于最近的质心。

Lloyd 算法：

1. 随机选取 K 个点作为初始质心
2. 将每个数据点分配给最近的质心
3. 将每个质心重新计算为其分配点的均值
4. 重复步骤 2-3 直到分配不再变化

目标函数（惯性，Inertia）衡量每个点到其所属质心的总平方距离。K 均值最小化这个值，但只能找到局部最小值。不同的初始化会产生不同的结果。

### 选择 K

两种标准方法：

**肘部法则（Elbow Method）：** 对 K = 1, 2, 3, ..., n 运行 K 均值。绘制惯性与 K 的关系图。找到"肘部"——即增加更多簇后惯性不再显著下降的拐点。

**轮廓系数（Silhouette Score）：** 对于每个点，衡量它与自身簇的相似度（a）与最近的其他簇的相似度（b）。轮廓系数为 (b - a) / max(a, b)，范围从 -1（分在了错误的簇）到 +1（聚类效果好）。所有点取平均即为全局分数。

### DBSCAN：基于密度的聚类

K 均值假设簇是球形的，并且需要你预先指定 K。DBSCAN（基于密度的聚类）则无需这两个假设。它将簇视为由稀疏区域分隔的密集区域。

两个参数：
- **eps**：邻域的半径
- **min_samples**：形成密集区域所需的最少点数

三类点：
- **核心点（Core Point）**：在 eps 距离内至少有 min_samples 个点
- **边界点（Border Point）**：在某个核心点的 eps 范围内，但自身不是核心点
- **噪声点（Noise Point）**：既不是核心点也不是边界点。这些就是异常值

DBSCAN 将彼此在 eps 范围内的核心点连接到同一个簇。边界点加入附近核心点的簇。噪声点不属于任何簇。

优势：能发现任意形状的簇，自动确定簇的数量，识别异常值。劣势：在密度差异较大的数据上效果不佳。

### 层次聚类

构建一棵聚类嵌套树（树状图，Dendrogram）。

凝聚型（自底向上）：
1. 将每个点视为独立的簇
2. 合并两个最近的簇
3. 重复直到只剩一个簇
4. 在所需的层级切割树状图以获得 K 个簇

簇之间的"近距离"可以用以下方式衡量：
- **单链接（Single Linkage）**：两个簇中任意两点之间的最小距离
- **全链接（Complete Linkage）**：任意两点之间的最大距离
- **平均链接（Average Linkage）**：所有点对之间的平均距离
- **Ward 方法**：导致总簇内方差增加最小的合并

### 高斯混合模型（GMM）

K 均值给出硬分配：每个点恰好属于一个簇。GMM 给出软分配：每个点有属于每个簇的概率。

GMM 假设数据由 K 个高斯分布的混合生成，每个分布有自己的均值和协方差。期望最大化算法（EM Algorithm）在以下两步之间交替：

- **E 步**：计算每个点属于每个高斯分布的概率
- **M 步**：更新每个高斯分布的均值、协方差和混合权重，以最大化数据的似然

GMM 可以建模椭圆形的簇（不仅限于 K 均值的球形），并能自然地处理重叠的簇。

### 何时使用哪种方法

| 方法 | 最佳场景 | 应避免的场景 |
|------|----------|-------------|
| K 均值 | 大数据集、球形簇、已知 K | 不规则形状、存在异常值 |
| DBSCAN | 未知 K、任意形状、异常检测 | 密度差异大、维度很高 |
| 层次聚类 | 小数据集、需要树状图、未知 K | 大数据集（O(n^2) 内存） |
| GMM | 重叠的簇、需要软分配 | 非常大的数据集、维度过多 |

### 基于聚类的异常检测

聚类天然支持异常检测（Anomaly Detection）：
- **K 均值**：远离所有质心的点即为异常
- **DBSCAN**：噪声点从定义上就是异常
- **GMM**：在所有高斯分布下概率都很低的点即为异常

## 动手实现

### 第 1 步：从零实现 K 均值

```python
import math
import random


def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def kmeans(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n_features = len(data[0])

    centroids = random.sample(data, k)

    for iteration in range(max_iterations):
        clusters = [[] for _ in range(k)]
        assignments = []

        for point in data:
            distances = [euclidean_distance(point, c) for c in centroids]
            nearest = distances.index(min(distances))
            clusters[nearest].append(point)
            assignments.append(nearest)

        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                new_centroids.append(random.choice(data))
                continue
            centroid = [
                sum(point[j] for point in cluster) / len(cluster)
                for j in range(n_features)
            ]
            new_centroids.append(centroid)

        if all(
            euclidean_distance(old, new) < 1e-6
            for old, new in zip(centroids, new_centroids)
        ):
            print(f"  Converged at iteration {iteration + 1}")
            break

        centroids = new_centroids

    return assignments, centroids
```

### 第 2 步：肘部法则与轮廓系数

```python
def compute_inertia(data, assignments, centroids):
    total = 0.0
    for point, cluster_id in zip(data, assignments):
        total += euclidean_distance(point, centroids[cluster_id]) ** 2
    return total


def silhouette_score(data, assignments):
    n = len(data)
    if n < 2:
        return 0.0

    clusters = {}
    for i, c in enumerate(assignments):
        clusters.setdefault(c, []).append(i)

    if len(clusters) < 2:
        return 0.0

    scores = []
    for i in range(n):
        own_cluster = assignments[i]
        own_members = [j for j in clusters[own_cluster] if j != i]

        if len(own_members) == 0:
            scores.append(0.0)
            continue

        a = sum(euclidean_distance(data[i], data[j]) for j in own_members) / len(own_members)

        b = float("inf")
        for cluster_id, members in clusters.items():
            if cluster_id == own_cluster:
                continue
            avg_dist = sum(euclidean_distance(data[i], data[j]) for j in members) / len(members)
            b = min(b, avg_dist)

        if max(a, b) == 0:
            scores.append(0.0)
        else:
            scores.append((b - a) / max(a, b))

    return sum(scores) / len(scores)


def find_best_k(data, max_k=10):
    print("Elbow method:")
    inertias = []
    for k in range(1, max_k + 1):
        assignments, centroids = kmeans(data, k)
        inertia = compute_inertia(data, assignments, centroids)
        inertias.append(inertia)
        print(f"  K={k}: inertia={inertia:.2f}")

    print("\nSilhouette scores:")
    for k in range(2, max_k + 1):
        assignments, centroids = kmeans(data, k)
        score = silhouette_score(data, assignments)
        print(f"  K={k}: silhouette={score:.4f}")

    return inertias
```

### 第 3 步：从零实现 DBSCAN

```python
def dbscan(data, eps, min_samples):
    n = len(data)
    labels = [-1] * n
    cluster_id = 0

    def region_query(point_idx):
        neighbors = []
        for i in range(n):
            if euclidean_distance(data[point_idx], data[i]) <= eps:
                neighbors.append(i)
        return neighbors

    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = region_query(i)

        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seed_set = list(neighbors)
        seed_set.remove(i)

        j = 0
        while j < len(seed_set):
            q = seed_set[j]

            if not visited[q]:
                visited[q] = True
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_samples:
                    for nb in q_neighbors:
                        if nb not in seed_set:
                            seed_set.append(nb)

            if labels[q] == -1:
                labels[q] = cluster_id

            j += 1

        cluster_id += 1

    return labels
```

### 第 4 步：高斯混合模型（EM 算法）

```python
def gmm(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n = len(data)
    d = len(data[0])

    indices = random.sample(range(n), k)
    means = [list(data[i]) for i in indices]
    variances = [1.0] * k
    weights = [1.0 / k] * k

    def gaussian_pdf(x, mean, variance):
        d = len(x)
        coeff = 1.0 / ((2 * math.pi * variance) ** (d / 2))
        exponent = -sum((xi - mi) ** 2 for xi, mi in zip(x, mean)) / (2 * variance)
        return coeff * math.exp(max(exponent, -500))

    for iteration in range(max_iterations):
        responsibilities = []
        for i in range(n):
            probs = []
            for j in range(k):
                probs.append(weights[j] * gaussian_pdf(data[i], means[j], variances[j]))
            total = sum(probs)
            if total == 0:
                total = 1e-300
            responsibilities.append([p / total for p in probs])

        old_means = [list(m) for m in means]

        for j in range(k):
            r_sum = sum(responsibilities[i][j] for i in range(n))
            if r_sum < 1e-10:
                continue

            weights[j] = r_sum / n

            for dim in range(d):
                means[j][dim] = sum(
                    responsibilities[i][j] * data[i][dim] for i in range(n)
                ) / r_sum

            variances[j] = sum(
                responsibilities[i][j]
                * sum((data[i][dim] - means[j][dim]) ** 2 for dim in range(d))
                for i in range(n)
            ) / (r_sum * d)
            variances[j] = max(variances[j], 1e-6)

        shift = sum(
            euclidean_distance(old_means[j], means[j]) for j in range(k)
        )
        if shift < 1e-6:
            print(f"  GMM converged at iteration {iteration + 1}")
            break

    assignments = []
    for i in range(n):
        assignments.append(responsibilities[i].index(max(responsibilities[i])))

    return assignments, means, weights, responsibilities
```

### 第 5 步：生成测试数据并运行所有算法

```python
def make_blobs(centers, n_per_cluster=50, spread=0.5, seed=42):
    random.seed(seed)
    data = []
    true_labels = []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_cluster):
            x = cx + random.gauss(0, spread)
            y = cy + random.gauss(0, spread)
            data.append([x, y])
            true_labels.append(label)
    return data, true_labels


def make_moons(n_samples=200, noise=0.1, seed=42):
    random.seed(seed)
    data = []
    labels = []
    n_half = n_samples // 2
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = math.cos(angle) + random.gauss(0, noise)
        y = math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
        labels.append(0)
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = 1 - math.cos(angle) + random.gauss(0, noise)
        y = 1 - math.sin(angle) - 0.5 + random.gauss(0, noise)
        data.append([x, y])
        labels.append(1)
    return data, labels


if __name__ == "__main__":
    centers = [[2, 2], [8, 3], [5, 8]]
    data, true_labels = make_blobs(centers, n_per_cluster=50, spread=0.8)

    print("=== K-Means on 3 blobs ===")
    assignments, centroids = kmeans(data, k=3)
    print(f"  Centroids: {[[round(c, 2) for c in cent] for cent in centroids]}")
    sil = silhouette_score(data, assignments)
    print(f"  Silhouette score: {sil:.4f}")

    print("\n=== Elbow Method ===")
    find_best_k(data, max_k=6)

    print("\n=== DBSCAN on 3 blobs ===")
    db_labels = dbscan(data, eps=1.5, min_samples=5)
    n_clusters = len(set(db_labels) - {-1})
    n_noise = db_labels.count(-1)
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    print("\n=== GMM on 3 blobs ===")
    gmm_assignments, gmm_means, gmm_weights, _ = gmm(data, k=3)
    print(f"  Means: {[[round(m, 2) for m in mean] for mean in gmm_means]}")
    print(f"  Weights: {[round(w, 3) for w in gmm_weights]}")
    gmm_sil = silhouette_score(data, gmm_assignments)
    print(f"  Silhouette score: {gmm_sil:.4f}")

    print("\n=== DBSCAN on moons (non-spherical clusters) ===")
    moon_data, moon_labels = make_moons(n_samples=200, noise=0.1)
    moon_db = dbscan(moon_data, eps=0.3, min_samples=5)
    n_moon_clusters = len(set(moon_db) - {-1})
    n_moon_noise = moon_db.count(-1)
    print(f"  Found {n_moon_clusters} clusters, {n_moon_noise} noise points")

    print("\n=== K-Means on moons (will fail to separate) ===")
    moon_km, moon_centroids = kmeans(moon_data, k=2)
    moon_sil = silhouette_score(moon_data, moon_km)
    print(f"  Silhouette score: {moon_sil:.4f}")
    print("  K-Means splits moons poorly because they are not spherical")

    print("\n=== Anomaly detection with DBSCAN ===")
    anomaly_data = list(data)
    anomaly_data.append([20.0, 20.0])
    anomaly_data.append([-5.0, -5.0])
    anomaly_data.append([15.0, 0.0])
    anomaly_labels = dbscan(anomaly_data, eps=1.5, min_samples=5)
    anomalies = [
        anomaly_data[i]
        for i in range(len(anomaly_labels))
        if anomaly_labels[i] == -1
    ]
    print(f"  Detected {len(anomalies)} anomalies")
    for a in anomalies[-3:]:
        print(f"    Point {[round(v, 2) for v in a]}")
```

## 实际使用

使用 scikit-learn，同样的算法只需一行代码：

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score as sklearn_silhouette

km = KMeans(n_clusters=3, random_state=42).fit(data)
db = DBSCAN(eps=1.5, min_samples=5).fit(data)
agg = AgglomerativeClustering(n_clusters=3).fit(data)
gmm_model = GaussianMixture(n_components=3, random_state=42).fit(data)
```

从零实现的版本让你清楚地看到这些库在做什么。K 均值在分配和重新计算之间迭代。DBSCAN 从密集种子点开始扩展聚类。GMM 在期望步和最大化步之间交替。库版本增加了数值稳定性、更智能的初始化（K-Means++）和 GPU 加速，但核心逻辑是一样的。

## 交付物

本课产出 K 均值、DBSCAN 和 GMM 的从零实现。聚类代码可以作为更高级无监督方法的基础复用。

## 练习

1. 实现 K-Means++ 初始化：不再随机选取质心，而是随机选取第一个，之后每个后续质心以与最近已有质心的平方距离成正比的概率选取。比较与随机初始化的收敛速度。
2. 将层次凝聚聚类添加到代码中。实现 Ward 链接并生成树状图（以嵌套合并列表的形式）。在不同层级切割并与 K 均值的结果进行比较。
3. 构建一个简单的异常检测流水线：在同一数据上运行 DBSCAN 和 GMM，标记两种方法一致认定的异常点（DBSCAN 中的噪声点、GMM 中的低概率点）。衡量重叠程度并讨论两种方法何时会产生分歧。

## 核心术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| 聚类（Clustering） | "把相似的东西分到一组" | 将数据划分为子集，使得组内相似度超过组间相似度，由特定的距离度量衡量 |
| 质心（Centroid） | "簇的中心" | 分配给某个簇的所有点的均值；K 均值用它作为簇的代表 |
| 惯性（Inertia） | "簇有多紧凑" | 每个点到其所属质心的平方距离之和；越低越紧凑 |
| 轮廓系数（Silhouette Score） | "簇的分离度有多好" | 对每个点，(b - a) / max(a, b)，其中 a 是簇内平均距离，b 是最近簇的平均距离 |
| 核心点（Core Point） | "密集区域中的点" | 在 DBSCAN 中，在 eps 距离内至少有 min_samples 个邻居的点 |
| EM 算法（EM Algorithm） | "软 K 均值" | 期望最大化：迭代计算归属概率（E 步）并更新分布参数（M 步） |
| 树状图（Dendrogram） | "聚类的树" | 显示层次聚类中簇合并顺序和距离的树形图 |
| 异常（Anomaly） | "离群值" | 不符合预期模式的数据点，被 DBSCAN 识别为噪声或被 GMM 识别为低概率 |

## 延伸阅读

- [Stanford CS229 - Unsupervised Learning](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) - Andrew Ng 关于聚类和 EM 的讲义
- [scikit-learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html) - 所有聚类算法的实用比较，附可视化示例
- [DBSCAN original paper (Ester et al., 1996)](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) - 提出基于密度聚类的原始论文
