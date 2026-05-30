# CS240 多视角序列排序算法实验说明

这份说明对应我们 `CS240_project.pdf` 里的算法部分：给定一组无序多视角图片和相机位姿，把每张图片当成图上的一个点，用图片/位姿相似度作为边权，然后比较不同的排序算法能不能生成更平滑的图片序列。

这个实验包是 CPU 友好的，不依赖 VSR 权重、不依赖 3DGS 训练，也不需要 GPU。它可以先用合成数据验证算法，后面也可以直接读取 NeRF Synthetic / Blender 格式的数据，例如 `lego`、`chair` 这类场景。

## 1. 我们完成了什么

新增代码主要在：

```text
cs240_ordering/
  __init__.py
  algorithms.py      # greedy, adaptive subsequences, Held-Karp DP, 2-opt, windowed DP
  data.py            # ViewSample 数据结构
  distances.py       # pose / feature / hybrid 距离矩阵
  loaders.py         # 读取 NeRF Synthetic / Blender 格式真实数据
  metrics.py         # 路径质量指标和运行时间/内存统计
  plotting.py        # 相邻边权曲线和序列可视化
  synthetic.py       # 合成测试数据和 greedy 反例

scripts/run_cs240_ordering_experiment.py
tests/test_cs240_ordering.py
```

核心目标不是重建 3D 模型，而是完成课程项目里的算法实验：

- 建图：每张图是一个 vertex，两张图之间的 dissimilarity 是 edge weight。
- 排序：找一条或多条序列，让相邻图片尽量相似。
- 比较：分析 greedy、DP、local search、adaptive subsequence 的质量和复杂度。

## 2. 环境准备

进入仓库根目录：

```bash
cd /storage/group/4dvlab/zym/CS240-Project
```

当前环境已经验证过可用：

```bash
python --version
python - <<'PY'
import numpy, cv2, PIL, matplotlib
print("numpy", numpy.__version__)
print("cv2", cv2.__version__)
print("PIL", PIL.__version__)
print("matplotlib", matplotlib.__version__)
PY
```

如果换到新环境，至少需要：

```bash
pip install numpy opencv-python Pillow matplotlib
```

运行单元测试：

```bash
python -m unittest discover -s tests
```

预期结果：

```text
Ran 6 tests ...
OK
```

## 3. 用合成数据跑完整实验

最简单命令：

```bash
python scripts/run_cs240_ordering_experiment.py
```

等价于：

```bash
python scripts/run_cs240_ordering_experiment.py \
  --data synthetic \
  --n 12 \
  --seed 240 \
  --distance all \
  --out output/cs240_ordering
```

常用参数：

```text
--data synthetic          使用合成数据
--n 12                    生成 12 张合成多视角图片
--seed 240                随机种子，保证结果可复现
--distance all            同时测试 pose / feature / hybrid 三种距离
--out output/...          输出目录
--dp-limit 16             N <= 16 时运行精确 Held-Karp DP
--window-size 8           windowed DP 的局部窗口大小
--thresholds 0.20 0.35 1.01
                           adaptive subsequence 的阈值，最后 1.01 保证覆盖所有图片
```

输出目录：

```text
output/cs240_ordering/
  metrics.csv             # 最重要的结果表
  orders.json             # 每个算法给出的具体排序
  cost_pose.npy           # pose 距离矩阵
  cost_feature.npy        # feature 距离矩阵
  cost_hybrid.npy         # hybrid 距离矩阵
  plots/                  # 相邻边权曲线
  strips/                 # 图片序列拼接图
```

## 4. 算法说明

### naive

直接按原始编号排序：

```text
0, 1, 2, 3, ...
```

这是最简单 baseline。如果数据本身已经按相机轨迹编号，naive 会很强；如果输入是无序的，naive 通常会很差。

### greedy nearest-neighbor

从第 0 张图开始，每一步选择当前图片最近的、还没访问过的图片：

```text
next = argmin C[current, unvisited]
```

优点：

- 简单；
- 快；
- 复杂度约为 `O(N^2)`。

缺点：

- 只看局部最优；
- 可能前面贪便宜，最后被迫接一条很大的跳边。

### adaptive subsequences

不强制把所有图片接成一条长序列，而是生成多条局部平滑子序列。

如果当前图片到最近未访问图片的距离大于阈值，就停止当前序列：

```text
if min C[current, unvisited] > threshold:
    start a new subsequence
```

意义：

- 更适合 VSR 这类需要局部连续输入的模型；
- 可以避免特别突兀的相邻帧；
- 但结果是多条序列，所以它的 `total_path_cost` 不能和单条完整路径算法完全等价比较。

### Held-Karp DP

精确动态规划，求小规模下的最优 Hamiltonian path：

```text
DP[S][j] = min_i DP[S - {j}][i] + C[i, j]
```

优点：

- 能得到全局最优；
- 可以作为 greedy 的 reference。

缺点：

- 时间复杂度 `O(N^2 2^N)`；
- 只适合小规模实验，所以默认 `N <= 16` 才跑。

### greedy + 2-opt

先用 greedy 得到一个序列，然后尝试反转中间某段。如果反转之后总成本下降，就接受。

它是一个 local search：

- 比 DP 快很多；
- 不保证全局最优；
- 可以修复一部分 greedy 的局部错误。

### greedy + windowed DP

先用 greedy 得到一条完整序列，再在局部窗口里用 DP 重新排序。

默认窗口大小：

```text
--window-size 8
```

它是 DP 和 greedy 的折中：

- 比全局 DP 更可扩展；
- 比纯 greedy 多一点局部全局性；
- 适合较大的真实场景。

## 5. 距离矩阵说明

### pose distance

使用相机中心距离和 viewing direction 角度差：

```text
pose_distance = camera_center_distance + viewing_direction_angle
```

适合相机轨迹比较规则的数据。

### feature distance

使用 ORB 特征匹配距离：

```text
feature_distance = average Hamming distance of matched ORB descriptors
```

适合图像内容本身能反映相邻关系的情况。

### hybrid distance

pose 和 feature 的加权组合：

```text
hybrid = 0.7 * normalized_pose + 0.3 * normalized_feature
```

这是默认推荐的折中方式。pose 提供几何稳定性，feature 提供图像内容信息。

## 6. 怎么看结果

打开：

```bash
less output/cs240_ordering/metrics.csv
```

每一行对应一种：

```text
distance type + algorithm
```

重要列：

```text
total_path_cost
```

整条序列所有相邻边权之和。越小表示整体越平滑。

```text
max_adjacent_jump
```

最大的一次相邻跳变。越小表示没有特别突兀的相邻帧。

```text
mean_adjacent_cost
```

平均相邻距离。越小表示平均过渡更平滑。

```text
std_adjacent_cost
```

相邻距离的标准差。越小表示序列过渡更均匀。

```text
coverage
```

覆盖率。`1.0` 表示所有图片都被使用。

```text
num_subsequences
```

子序列数量。普通单序列算法通常是 `1`；adaptive 可能大于 `1`。

```text
runtime_seconds
```

运行时间。DP 通常会明显慢于 greedy。

```text
peak_memory_bytes
```

Python 层统计到的峰值内存。DP 会更高。

查看具体排序：

```bash
less output/cs240_ordering/orders.json
```

例如：

```json
"greedy": [0, 1, 10, 7, 8, 5, 4, 3, 6, 9, 11, 2]
```

表示 greedy 认为图片应该按这个顺序送入后续 video-style 模型。

查看曲线：

```text
output/cs240_ordering/plots/
```

曲线的 y 值是相邻帧 edge cost。好的序列通常应该：

- 总体低；
- 没有很高的尖峰；
- 波动小。

查看序列拼图：

```text
output/cs240_ordering/strips/
```

这些图可以直观看排序后相邻图片是否平滑。

## 7. 这次合成数据结果怎么解释

我们已经跑过：

```bash
python scripts/run_cs240_ordering_experiment.py --n 12 --seed 240
```

结果大致是：

```text
pose:
  naive / greedy / DP 都得到相同成本 0.2256

feature:
  naive       4.5826
  greedy      3.2218
  Held-Karp   2.8967
  adaptive    0.5691, 但产生 7 条子序列

hybrid:
  naive / greedy 1.4826
  Held-Karp      1.3400
  adaptive       0.5875, 但产生 4 条子序列
```

解释：

- 在 `pose` 距离下，合成相机本来就是按圆形轨迹编号的，所以 `0,1,2,...` 已经很好。
- 在 `feature` 距离下，图片内容相似性和编号顺序不完全一致，所以 greedy 比 naive 好，DP 又比 greedy 更好。
- 在 `hybrid` 距离下，pose 信息比较强，greedy 已经不错，但 DP 仍然能找到更低成本。
- adaptive 的总成本很低，是因为它允许断开大跳变，生成多条局部平滑子序列。这个结果适合说明“局部平滑 vs 单序列覆盖”的 trade-off。

报告里可以这么写：

```text
Greedy nearest-neighbor is efficient and usually improves over naive ordering, but it is not globally optimal.
Held-Karp DP gives the optimal reference on small scenes, showing the optimality gap of greedy.
Adaptive subsequence construction reduces large adjacent jumps by splitting one long path into several smooth local paths.
```

## 8. 用真实 NeRF Synthetic / Blender 数据运行

这个仓库本身没有包含完整 NeRF Synthetic 数据集。官方 README 也要求另外下载数据，例如 NeRF Synthetic 的 `lego`、`chair` 等场景。

真实数据目录通常长这样：

```text
/path/to/nerf_synthetic/lego/
  transforms_train.json
  transforms_val.json
  transforms_test.json
  train/
    r_0.png
    r_1.png
    ...
  val/
  test/
```

如果你已经有这种目录，可以直接运行：

```bash
python scripts/run_cs240_ordering_experiment.py \
  --data blender \
  --source /path/to/nerf_synthetic/lego \
  --split train \
  --limit 16 \
  --distance all \
  --out output/cs240_ordering_lego
```

参数说明：

```text
--data blender
```

表示读取 NeRF Synthetic / Blender 格式，不使用合成数据。

```text
--source /path/to/scene
```

场景目录，比如 `.../lego` 或 `.../chair`，不是整个数据集根目录。

```text
--split train
```

读取 `transforms_train.json` 和 `train/` 图片。

```text
--limit 16
```

只取前 16 张图。建议一开始先用小数量，因为 Held-Karp DP 很慢。确认流程跑通后可以加大。

如果想在更多图片上跑，但不想跑全局 DP：

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

这里 `N=80` 时不会跑 Held-Karp 全局 DP，因为超过 `--dp-limit 12`。仍然会跑：

- naive
- greedy
- adaptive
- greedy + 2-opt
- greedy + windowed DP

## 9. 如果使用 SequenceMatters 的 LR/VSR 数据

SequenceMatters 的训练流程会生成或使用类似：

```text
scene/
  transforms_train.json
  train/
  train_lr/
```

当前排序实验读取的是 `transforms_train.json` 里 `file_path` 对应的图片。也就是说，如果 `file_path` 是 `train/r_0`，它会读取：

```text
scene/train/r_0.png
```

如果你想对 LR 图像排序，有两种办法：

1. 直接把 LR 图片放在 `train/` 下，保留 `transforms_train.json`。
2. 新建一个轻量场景目录，把 `transforms_train.json` 复制过去，并让 `train/` 指向或保存 LR 图片。

示例：

```text
my_lego_lr_for_ordering/
  transforms_train.json
  train/
    r_0.png
    r_1.png
```

然后运行：

```bash
python scripts/run_cs240_ordering_experiment.py \
  --data blender \
  --source /path/to/my_lego_lr_for_ordering \
  --split train \
  --limit 32 \
  --distance hybrid \
  --out output/cs240_ordering_lego_lr
```

## 10. 推荐实验组合

为了写报告，建议至少跑这些：

合成数据：

```bash
python scripts/run_cs240_ordering_experiment.py \
  --data synthetic \
  --n 12 \
  --seed 240 \
  --distance all \
  --out output/cs240_ordering_synthetic
```

真实 `lego` 小子集：

```bash
python scripts/run_cs240_ordering_experiment.py \
  --data blender \
  --source /path/to/nerf_synthetic/lego \
  --split train \
  --limit 16 \
  --distance all \
  --out output/cs240_ordering_lego_16
```

真实 `chair` 小子集：

```bash
python scripts/run_cs240_ordering_experiment.py \
  --data blender \
  --source /path/to/nerf_synthetic/chair \
  --split train \
  --limit 16 \
  --distance all \
  --out output/cs240_ordering_chair_16
```

真实较大子集，不跑全局 DP：

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

## 11. 报告中可以怎么组织分析

建议按下面结构写：

1. Problem formulation

```text
We model unordered multi-view images as a complete weighted graph. Each view is a vertex, and each edge weight measures view dissimilarity. The goal is to construct a Hamiltonian-path-like ordering with low adjacent transition cost.
```

2. Distance definitions

说明 pose、feature、hybrid 三种 edge weight。

3. Algorithm comparison

比较：

- naive ordering
- greedy nearest-neighbor
- adaptive subsequences
- Held-Karp DP
- greedy + 2-opt
- greedy + windowed DP

4. Metrics

使用：

- total path cost
- max adjacent jump
- mean adjacent cost
- coverage
- runtime
- memory

5. Main observations

可以从结果里总结：

- Greedy 很快，通常优于 naive。
- DP 小规模最优，但复杂度高。
- Local search/windowed DP 是速度和质量之间的折中。
- Adaptive subsequences 能减少大跳变，但会输出多条序列。
- 不同 distance definition 会改变排序行为。

## 12. 常见问题

### 为什么 adaptive 的 total_path_cost 特别低？

因为 adaptive 允许断开序列。它不是强制把所有图片接成一条长链，而是生成多条局部平滑序列。因此它少计算了那些跨子序列的大跳边。

分析 adaptive 时应该重点看：

```text
max_adjacent_jump
num_subsequences
coverage
```

不要只看 `total_path_cost`。

### 为什么 pose 下 naive 和 greedy 一样？

如果数据本身就是沿着相机轨迹按顺序编号的，那么 naive 已经是很好的顺序。真实无序输入里，可以先打乱文件顺序，或使用 feature/hybrid 距离观察算法差异。

### 为什么 DP 很慢？

Held-Karp DP 的复杂度是：

```text
O(N^2 2^N)
```

所以它只适合小场景或小子集。大场景建议使用 greedy、2-opt 或 windowed DP。

### GPU 要用吗？

这个算法实验不需要 GPU。VSR/3DGS 才需要 GPU。

### 计算节点没网怎么办？

本实验不需要网络。只要 Python 依赖已经装好，合成数据和本地真实数据都可以直接跑。

## 13. 一键检查命令

```bash
cd /storage/group/4dvlab/zym/CS240-Project

python -m unittest discover -s tests

python scripts/run_cs240_ordering_experiment.py \
  --data synthetic \
  --n 12 \
  --seed 240 \
  --distance all \
  --out output/cs240_ordering
```

如果这两条都成功，就说明算法实验包和结果生成流程是正常的。
