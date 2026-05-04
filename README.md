# Traditional SfM — 手动实现的增量式运动恢复结构

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个基于视觉的增量式 **Structure from Motion (SfM)** 三维重建管线。对极几何、PnP、三角化、光束法平差等核心算法手动实现，特征提取与 BA 优化借助 OpenCV / SciPy 加速。

## 项目结构

```
traditional/
├── __init__.py                  # 包声明
├── config.py                    # 相机内参 & 畸变系数的 JSON 读写
├── calibration.py               # 去畸变（Brown-Conrady 模型）
├── features.py                  # 特征提取（OpenCV SIFT，手动 DoG 回退）
├── matching.py                  # 特征匹配（L2 暴力 + Lowe ratio test）
├── geometry.py                  # 对极几何（归一化 8 点法 → F/E → RANSAC）
├── pose.py                      # 位姿恢复（E 分解 → 4 候选 → 正深度检验）
├── triangulation.py             # DLT 线性三角化 + 重投影误差过滤
├── sfm.py                       # 增量式 SfM（两视图初始化 + EPNP PnP + 动态参考帧）
├── bundle_adjustment.py         # BA（SciPy TRF + 解析雅可比 + Huber loss）
├── utils.py                     # PLY 导出 + Open3D 可视化
├── pipeline.py                  # 主流程编排入口
├── tests/
│   └── test_synthetic.py        # 合成数据测试（定量评估：位姿 / 重投影 / 3D 点误差）
├── scripts/
│   ├── run_data_pipeline.py     # 自写管线启动脚本
├── config/
│   └── camera.json              # 相机参数 JSON
├── data_image/                  # 图片数据集
├── output/                      # 输出 PLY 点云
└── docs/agents/                 # Agent skills 配置
```

## 快速开始

### 1. 环境

```bash
pip install numpy pillow scipy
pip install opencv-python        # SIFT + EPNP PnP
pip install open3d               # 可选，点云可视化
```

### 2. 准备相机参数

```json
{
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "distortion": [k1, k2, p1, p2, k3],
  "width": 2272,
  "height": 1704
}
```

焦距可从 EXIF 推算：`fx = focal_mm × width / sensor_width_mm`

### 3. 运行

```bash
# 合成数据测试（验证几何正确性）
python tests/test_synthetic.py

# 真实图片重建
python scripts/run_data_pipeline.py \
    --image-dir data_image/Dog_RGB \
    --config config/camera.json \
    --output-dir output/dog

# COLMAP 对比
python scripts/reconstruct_colmap.py
```

## 算法概览

### 特征提取与匹配

| 步骤 | 方法 |
|------|------|
| 特征检测 | OpenCV `cv2.SIFT_create()` 优先；手动 DoG + SIFT-like 128D 描述子回退 |
| 特征匹配 | L2 暴力搜索 + Lowe ratio test (0.75) + 交叉验证 |

### 增量式 SfM (`sfm.py`)

| 阶段 | 方法 | 说明 |
|------|------|------|
| 两视图初始化 | RANSAC + 8 点法 → F → E → 4 候选 (R,t) | 正深度检验选最优 |
| PnP 位姿估计 | **OpenCV `SOLVEPNP_EPNP`** (优先) / DLT 回退 | EPNP 最少 4 点，抗噪声 |
| 三角化 | DLT 线性三角化 + 角度/深度/重投影过滤 |
| 参考帧管理 | **动态切换**：匹配 < 50 时自动更新 | 解决 360° 序列的漂移问题 |
| 初始对选择 | 基线约束：7%~33% 序列长度 + ≥50 匹配 | 避免窄基线三角化退化 |

### 光束法平差 (`bundle_adjustment.py`)

| 组件 | 实现 |
|------|------|
| 求解器 | **SciPy `least_squares`** (Trust Region Reflective) |
| 雅可比 | **解析导数**（Rodrigues + 投影链式法则），稀疏矩阵 |
| 鲁棒核 | Huber loss |
| 参数化 | 轴角旋转 (3) + 平移 (3) + 3D 点 (3) |

### 对极几何 (`geometry.py`)

- **归一化 8 点法**：Hartley 归一化 → SVD → 秩-2 约束 → 去归一化
- **RANSAC**：Sampson 距离，自适应迭代次数
- **本质矩阵**：`E = K₂ᵀFK₁`，SVD 强制奇异值 `[1,1,0]`

### 位姿恢复 (`pose.py`)

- E 矩阵 SVD 分解得 4 组 (R,t) 候选
- 三角化所有点后正深度检验，选最多前方点的那组

### 三角化 (`triangulation.py`)

- DLT 线性解：构造 4×4 齐次线性系统 A·X=0
- 重投影误差 > 4px（合成）/ 10px（真实）的点被过滤

## 合成数据测试结果

| 指标 | SfM | BA 后 |
|------|-----|-------|
| 注册率 | 5/5 (100%) | 5/5 (100%) |
| 旋转误差 | 0.00° | 0.00° |
| 重投影误差 | 0.00 px | 0.00 px |
| 3D 点 | 167 | 167 |

## 已知局限

1. **单目尺度不确定性**：重建仅有相对尺度，需外部参照定绝对尺度
2. **相机标定**：需外部提供内参，不支持自动标定
3. **稠密重建**：仅输出稀疏点云，无 MVS 稠密化
4. **纯旋转退化**：8 点法在纯旋转 / 平面场景下退化为多解
5. **滚动快门**：手机视频帧可能存在滚动快门畸变

## License

MIT
