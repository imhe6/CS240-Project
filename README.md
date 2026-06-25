# CS 240 Project (26Spring) - Multi-View Sequence Ordering

此仓库是我们所参考的项目 [SequenceMatters](https://github.com/DHPark98/SequenceMatters)（AAAI 2025）的仓库，我们在阅读此项目的内容后，将算法部分（即选择图片路径的算法）单独进行了编写和实验。
本仓库分为两部分：

- **`cs240_ordering/` 目录**（我们的工作）：从 SequenceMatters 抽离并重新编写了的多视角图片序列排序算法模块。将无序多视角图片建模为图上的 TSP/路径问题，用图片相似度与位姿距离等因素作为边权，比较 greedy、DP、2-opt、adaptive subsequence 等排序策略的效果。

    详细说明见 [CS240_ORDERING_README.md](CS240_ORDERING_README.md)。
- **其余代码**：原 [SequenceMatters](https://github.com/DHPark98/SequenceMatters)（AAAI 2025）项目代码，保留作为参考。为顺利运行其实验而对其代码（主要是配置文件）进行了必要的修改，项目本身的算法等的相关代码未做修改。

    原项目 README 见 [ORIGINAL_README.md](ORIGINAL_README.md)。

## Quick Start

```bash
cd SequenceMatters
pip install numpy opencv-python Pillow matplotlib

# 运行实验（使用默认参数）
python scripts/run_cs240_ordering_experiment.py
```