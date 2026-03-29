<div align="center">

# 基于视觉的三维重建  
### Vision-based 3D Reconstruction

> 基于图像 / 视频的目标三维重建项目，包含传统方法版与深度学习方法版。  
> 输出三维点云、重建模型，并针对尺度模糊、纹理缺失等问题进行分析与优化。

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red.svg" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-green.svg" />
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey.svg" />
  <img src="https://img.shields.io/badge/GPU-RTX%205090-orange.svg" />
</p>

</div>

---

## 🔍 重建效果预览

<!--
【备注】
1. 这里强烈建议放一张“项目首页大图”，观感提升非常明显。
2. 最推荐放一张横向拼图，例如：
   左：输入图像
   中：稀疏点云
   右：稠密点云 / Mesh
3. 文件建议放在 docs/assets/preview.png
4. 如果你后面有更好的结果，也可以替换成 GIF。
-->

<p align="center">
  <img src="docs/assets/preview.png" width="90%" alt="reconstruction preview"/>
</p>

<!--
【可替换方案】
如果你后面想改成 GIF，可以这样写：

<p align="center">
  <img src="docs/assets/demo.gif" width="85%" alt="3d reconstruction demo"/>
</p>
-->

---

## ✨ 项目简介

<!--
【备注】
这里写 1~2 段就够，不要太长。
建议回答这几个问题：
1. 这是个什么项目？
2. 项目要解决什么问题？
3. 最终输出什么结果？
4. 为什么分成传统方法版和深度学习方法版？
-->

本项目围绕“**基于视觉的三维重建**”展开，旨在利用图像或视频输入恢复目标的三维结构，输出稀疏点云、稠密点云及三维模型。

根据课程要求，项目包含两个相互独立、均可运行的实现版本：

- **传统方法版**：手动实现核心计算机视觉算法，体现对经典三维重建流程的理解。
- **深度学习方法版**：基于最新相关大模型进行改进，并完成 SFT 微调或 LoRA 适配。

---

## 🌟 项目亮点

<!--
【备注】
这里适合写“这个项目有什么值得一看”的点。
建议控制在 4~6 条。
-->

- 双版本独立实现，覆盖传统几何方法与深度学习方法
- 支持手机拍照 / 多视图图像输入
- 支持从图像采集、预处理到三维模型输出的完整流程
- 可输出稀疏点云、稠密点云与三维网格模型
- 包含实验对比、误差分析与可视化展示

---

## 🧠 方法总览

<!--
【备注】
这里建议放一张流程图，非常加分。
例如：
数据采集 → 相机标定 → 特征提取 → 特征匹配 → SfM → BA → MVS → 输出模型
如果后面有深度学习路线图，也可以再放一张。
-->

<p align="center">
  <img src="docs/assets/pipeline.png" width="92%" alt="pipeline"/>
</p>

### 传统方法版
<!--
【备注】
这里先写流程概述，不要一开始写太细。
-->
1. 相机标定与去畸变  
2. 特征提取与特征匹配  
3. 本质矩阵 / 基础矩阵估计  
4. 两视图初始化  
5. 增量式 SfM 重建  
6. Bundle Adjustment 优化  
7. 稠密重建（可选）  
8. 点云 / 网格模型输出  

### 深度学习方法版
<!--
【备注】
这里写你们后续深度学习版的总路线。
后面你填具体模型名、改进点、微调方式。
-->
1. 选择基础大模型  
2. 构建训练 / 微调数据  
3. 设计模型改进策略  
4. 进行 SFT / LoRA 微调  
5. 执行推理与三维重建输出  
6. 与传统方法进行对比分析  

---

## 🗂️ 项目结构

<!--
【备注】
这里尽量和你的实际仓库结构一致。
如果后面目录改了，这里也要同步。
-->

```bash
.
├── README.md
├── docs/                    # 文档、开题报告、实验记录、结果截图
├── data/                    # 数据集
│   ├── raw/
│   ├── processed/
│   └── splits/
├── configs/                 # 配置文件
├── scripts/                 # 训练、推理、评测、可视化脚本
├── outputs/                 # 输出结果：点云、模型、日志等
├── checkpoints/             # 模型权重
├── traditional/             # 传统方法版
│   ├── calibrate/
│   ├── features/
│   ├── matching/
│   ├── geometry/
│   ├── sfm/
│   ├── mvs/
│   └── utils/
├── deep_learning/           # 深度学习方法版
│   ├── datasets/
│   ├── models/
│   ├── trainer/
│   ├── inference/
│   ├── lora/
│   └── utils/
└── requirements.txt