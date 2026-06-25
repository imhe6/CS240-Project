# CS240 Project: Multi-View Sequence Ordering

> 本模块为课程项目算法部分的核心代码，独立于原 SequenceMatters 项目编写与实验。

## Overview

给定一组无序多视角图像及其相机位姿，将每张图像建模为图上的节点，以图像/位姿相似度作为边权，比较不同排序算法生成平滑图像序列的效果。

## Code Structure

主要实现如下：

```text
cs240_ordering/
  algorithms.py      # greedy, adaptive subsequences, Held-Karp DP, 2-opt, windowed DP
  data.py            # ViewSample 数据结构
  distances.py       # pose / feature / hybrid 距离矩阵
  loaders.py         # NeRF Synthetic / Blender 格式数据读取
  metrics.py         # 路径质量指标与资源统计
  plotting.py        # 相邻边权曲线与序列可视化
  synthetic.py       # 合成测试数据与 greedy 反例

scripts/run_cs240_ordering_experiment.py   # 实验入口
tests/test_cs240_ordering.py               # Unittest
```

核心任务：

- **建图**：每张图像为一个节点，节点间的 dissimilarity 为边权。
- **排序**：寻找一条或多条序列，使相邻图像尽可能相似。
- **评估**：对比 greedy、DP、local search、adaptive subsequence 等算法的质量与复杂度。

## Environment

依赖：

```bash
pip install numpy opencv-python Pillow matplotlib
```

验证所用 Unittest：

```bash
python -m unittest discover -s tests
```

预期输出：`Ran 6 tests ... OK`

## Usage

### 1. 合成 (Synthetic) 数据实验

通过代码实时随机生成数个固定物体，生成其在若干不同视角下的图片，然后用这些生成的图片进行实验。

```bash
python scripts/run_cs240_ordering_experiment.py
```

其默认参数如下，等价于：

```bash
python scripts/run_cs240_ordering_experiment.py \
  --data synthetic \
  --n 12 \
  --seed 240 \
  --distance all \
  --out output/cs240_ordering
```

参数说明：

| 参数 | 含义 |
|------|------|
| `--data synthetic` | 使用合成数据 |
| `--n 12` | 生成 12 张合成多视角图像；如果是真是数据集 |
| `--seed 240` | 随机种子 |
| `--distance all` | 指定所用的距离指标，即 pose / feature / hybrid 。设为 `all` 即测试全部三个指标 |
| `--out output/...` | 输出目录 |
| `--dp-limit 16` | 设置运行 Held-Karp DP 的数据集大小阈值：图片数量小于此才会运行（因为其运行时间为指数级，图片太多会很慢，见后文） |
| `--window-size 8` | Windowed DP 方法的窗口大小 |
| `--thresholds 0.20 0.35 1.01` | Adaptive subsequence 阈值 |

### 2. 真实数据实验

使用实际的数据集（若干实际物体的多视角图片）进行实验。

与原项目 (SequenceMatters) 的类似，需准备 NeRF Synthetic / Blender 格式的场景数据，目录结构如下：

```text
/path/to/nerf_synthetic/{item_name}/
  transforms_train.json
  train/
    r_0.png, r_1.png, ...
```

实验运行示例：

```bash
python scripts/run_cs240_ordering_experiment.py \
  --data blender \
  --source /path/to/nerf_synthetic/lego \
  --split train \
  --limit 16 \
  --distance all \
  --out output/cs240_ordering_lego_16
```
其中 `--limit` 参数用于指定对每个物品的数据最多采集其多少张图片。`--limit 16` 表明最多采取前 16 张，其余图片会被忽略。

在大规模数据上，会跳过全局 Held-Karp DP 不运行（因为其运行时间为指数级，图片太多会很慢）。
默认为 16，但可通过 dp-limit 参数指定：

```bash
python scripts/run_cs240_ordering_experiment.py \
  --data blender \
  --source /path/to/nerf_synthetic/lego \
  --split train \
  --limit 80 \
  --distance hybrid \
  --dp-limit 12 \
  --window-size 8 \
  --out output/cs240_ordering_lego_80
```

### 输出结果示例

```text
output/cs240_ordering/
  metrics.csv       # 核心结果
  orders.json       # 各算法给出的具体排序
  cost_{distance_metric}.npy        # 距离矩阵
  plots/            # 相邻边权曲线
  strips/           # 图像序列拼接图
```

## Algorithms

### 1. Naive

按原始编号排列 $0, 1, 2, \ldots$。若数据本身按相机轨迹编号，该 baseline 即接近最优；若输入无序则性能较差。

### 2. Greedy Nearest-Neighbor

从起点出发，每次选取距离当前节点最近的未访问节点：

$$\text{next} = \arg\min_{v \notin \text{visited}} C[\text{current}, v]$$

- 复杂度 $O(N^2)$
- 缺点：贪心局部最优可能累积为全局次优

### 3. Adaptive Subsequences

不强制连接所有图像为单条序列，当最近未访问节点距离超过阈值时另起子序列：

$$\text{if } \min_{v \notin \text{visited}} C[\text{current}, v] > \theta \text{ then start new subsequence}$$

- 适合需局部连续输入的 VSR 类模型
- 避免突兀的相邻帧跳变
- 结果为多条序列，`total_path_cost` 与其他单序列算法不可直接比较

### 4. Held-Karp DP

精确动态规划，求解小规模最优 Hamiltonian path：

$$DP[S][j] = \min_{i \in S \setminus \{j\}} DP[S \setminus \{j\}][i] + C[i, j]$$

- 可得全局最优，作为 greedy 的 reference
- 复杂度 $O(N^2 2^N)$，仅适用于小规模（默认 $N \leq 16$）

### 5. Greedy + 2-opt

在 greedy 结果基础上，通过反转子段进行 local search，接受成本下降的改进。

- 比 DP 快，不保证全局最优
- 可修复部分 greedy 的局部错误

### 6. Greedy + Windowed DP

先用 greedy 得到完整序列，再在局部窗口内用 DP 重排（默认窗口大小 8）。

- DP 与 greedy 的折中
- 比全局 DP 可扩展，比纯 greedy 更全局
- 适合大规模真实场景

## Distance Metrics

### 1. Pose Distance

基于相机中心距离与 viewing direction 角度差：

$$\text{pose\_dist} = \text{center\_dist} + \text{angle\_diff}$$

适合相机轨迹规则的数据。

### 2. Feature Distance

基于 ORB 特征匹配的 Hamming 距离，反映图像内容相似性。

### 3. Hybrid Distance

$$\text{hybrid} = 0.7 \times \text{norm\_pose} + 0.3 \times \text{norm\_feature}$$

默认推荐的折中方案：pose 提供几何稳定性，feature 提供内容信息。

## Evaluation Metrics

`metrics.csv` 中每行为一种 `(distance, algorithm)` 组合，关键列：

| 列 | 含义 |
|----|------|
| `total_path_cost` | 序列所有相邻边权之和，越小越平滑 |
| `max_adjacent_jump` | 最大相邻跳变，越小越好 |
| `mean_adjacent_cost` | 平均相邻距离 |
| `std_adjacent_cost` | 相邻距离标准差，越小越均匀 |
| `coverage` | 覆盖率，1.0 表示所有图像均被使用 |
| `num_subsequences` | 子序列数（单序列算法为 1） |
| `runtime_seconds` | 运行时间 |
| `peak_memory_bytes` | 峰值内存 |

排序结果见 `orders.json`，如 `"greedy": [0, 1, 10, 7, 8, ...]`。

可视化：`plots/` 为相邻边权曲线（y 值为 edge cost，好的序列总体低、无尖峰、波动小），`strips/` 为图像序列拼接图。

## Example Results (Synthetic, n=12, seed=240)

```bash
python scripts/run_cs240_ordering_experiment.py --n 12 --seed 240
```

| Distance | Naive | Greedy | Held-Karp | Adaptive |
|----------|-------|--------|-----------|----------|
| Pose | 0.2256 | 0.2256 | 0.2256 | — |
| Feature | 4.5826 | 3.2218 | 2.8967 | 0.5691 (7 subseqs) |
| Hybrid | 1.4826 | 1.4826 | 1.3400 | 0.5875 (4 subseqs) |

**分析**：

- Pose 距离下合成相机按圆形轨迹编号，naive 已是最优。
- Feature 距离下图像内容相似性与编号不一致，greedy 优于 naive，DP 优于 greedy。
- Hybrid 距离下 pose 信号较强，greedy 已接近最优，DP 仍有小幅改进。
- Adaptive 的总成本显著更低，因其允许断开大跳变——体现了"局部平滑 vs. 单序列覆盖"的 trade-off。

## 8. Recommended Experiment Configurations

| 数据 | 命令 |
|------|------|
| 合成 | `--data synthetic --n 12 --seed 240 --distance all` |
| Lego (16) | `--data blender --source .../lego --limit 16 --distance all` |
| Chair (16) | `--data blender --source .../chair --limit 16 --distance all` |
| Lego (80, 无 DP) | `--data blender --source .../lego --limit 80 --distance hybrid --dp-limit 12` |

## 常见问题

**Adaptive 的 `total_path_cost` 偏低**：Adaptive 生成多条子序列，不计跨子序列的大跳边，因此 `total_path_cost` 偏低；和其他单序列算法比较时，建议同时考虑其他 metrics。

**Pose 距离下 naive 与 greedy 结果相同**：少数情况下，数据原先就已按相机轨迹顺序编号，导致 naive 已是最优解；可以手动打乱顺序或改用 feature/hybrid 距离。

**Held-Karp DP 很慢**：Held-Karp DP 的复杂度是 $O(N^2 2^N)$，因此编写代码时设置了一个阈值，如数据量（图片数）超出此阈值则不在其上使用 Held-Karp DP。默认仅适用于 $N \leq 16$ 的数据集（`--dp-limit 16`）；数据较大的场景建议使用 greedy、2-opt 或 windowed DP。

**是否需要 GPU**：不需要，所有算法均为基本的 CPU 实现。

