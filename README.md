# Traditional SfM — 手动实现的增量式运动恢复结构

一个基于视觉的增量式 **Structure from Motion (SfM)** 三维重建管线，核心算法全部手动实现（仅使用 `numpy` / `scipy` 做基础运算），不依赖 OpenCV 高级 API。

---

## 项目结构

```
traditional/
├── __init__.py              # 包声明，统一导出所有公共接口
├── config.py                # 相机内参 & 畸变系数的 JSON 配置读写
├── calibration.py           # 去畸变：径向畸变 + 切向畸变 + 双线性插值
├── features.py              # 特征提取：DoG 极值检测 + SIFT-like 128D 描述子
├── matching.py              # 特征匹配：L2 暴力匹配 + Lowe ratio test + 交叉验证
├── geometry.py              # 对极几何：归一化 8 点法 → F/E 矩阵 → RANSAC (Sampson)
├── pose.py                  # 位姿恢复：E 矩阵 SVD 分解 → 4 候选 (R,t) → 正深度检验
├── triangulation.py         # 三角化：DLT 线性三角化 + 重投影误差过滤
├── sfm.py                   # 增量式 SfM：两视图初始化 + 增量 PnP + Track 管理
├── bundle_adjustment.py     # 光束法平差：稀疏 LM + Schur 补 + Huber 鲁棒核
├── utils.py                 # 工具：PLY 点云导出 + Open3D 点云/相机可视化
├── pipeline.py              # 主流程编排入口
├── test_synthetic.py        # 合成数据测试（生成 → 投影 → 重建验证）
└── README.md                # 本文件
```

---

## 运行方式

### 1. 环境依赖

```bash
pip install numpy pillow imageio
pip install open3d          # 可选，用于点云可视化
```

> 只需 `numpy` 即可运行核心管线。`pillow` / `imageio` 用于图片读取（任一即可），`open3d` 用于交互式点云可视化。

### 2. 准备相机配置文件

在任意路径创建 `camera.json`：

```json
{
  "K": [[800.0, 0.0, 400.0], [0.0, 800.0, 300.0], [0.0, 0.0, 1.0]],
  "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
  "width": 800,
  "height": 600
}
```

| 字段 | 说明 |
|------|------|
| `K` | 3×3 内参矩阵 `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]` |
| `distortion` | 畸变系数 `[k1, k2, p1, p2, k3]` |
| `width` / `height` | 可选，图像尺寸 |

### 3. 运行重建

```python
from traditional.pipeline import run_pipeline

result = run_pipeline(
    image_dir="path/to/images/",     # 多视角图片目录
    config_path="path/to/camera.json",
    output_dir="output/",          # 可选，输出目录
    skip_undistort=False,           # 可选，跳过去畸变
)

# 返回: { 'cameras': [...], 'points3d': np.ndarray, 'ply_path': str }
```

### 4. 运行合成数据测试

```bash
python traditional/test_synthetic.py
```

测试流程：生成 200 个随机 3D 点 → 5 个虚拟相机投影 → 特征匹配 → 对极几何 → 增量 SfM → BA 优化 → PLY 导出。

---

## 核心算法实现说明

所有算法均由 `numpy` 基础运算手动实现，未使用 OpenCV 的 `cv2.findFundamentalMat`、`cv2.triangulatePoints`、`cv2.solvePnP` 等高级接口。

### 1. 去畸变 (`calibration.py`)

| 步骤 | 方法 |
|------|------|
| 畸变模型 | Brown-Conrady 模型：径向畸变 `k1,k2,k3` + 切向畸变 `p1,p2` |
| 矫正方式 | 逆映射：对每个输出像素 `(u,v)`，计算其在原始畸变图像中的对应坐标 `(x_d, y_d)` |
| 插值 | 双线性插值（手动实现） |

**核心公式**：
```
x_n = (u - cx) / fx
y_n = (v - cy) / fy
r² = x_n² + y_n²
x_d = x_n * (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x_n*y_n + p2*(r² + 2*x_n²)
y_d = y_n * (1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2*y_n²) + 2*p2*x_n*y_n
```

---

### 2. 特征提取 (`features.py`)

#### DoG (Difference of Gaussian) 极值检测

| 步骤 | 实现细节 |
|------|---------|
| 高斯金字塔 | `_build_gaussian_pyramid`：octave 间降采样 ×2，每个 octave 内 n_scales+3 层 |
| DoG 金字塔 | 相邻高斯层差分 `_build_dog_pyramid` |
| 极值检测 | 3×3×3 邻域非极大值抑制 `_detect_extrema` |
| 亚像素精化 | 二阶 Taylor 展开插值 `_refine_keypoint`（最多 5 次迭代） |
| 边缘抑制 | Hessian 矩阵曲率比 `_check_edge`（阈值 `r=10`） |
| 可分离卷积 | `_convolve1d_separable`：手动实现 separable convolution（避免 scipy 依赖） |

#### SIFT-like 描述子

| 步骤 | 参数 | 说明 |
|------|------|------|
| 主方向 | 36 bins 梯度直方图 | 峰值 80% 以上产生多个方向 |
| 描述子区域 | 16×16 patch | 以主方向旋转变换 |
| 空间网格 | 4×4 sub-regions | 每个 sub-region 8 方向 |
| 描述子维度 | **128 维** (4×4×8) | 梯度幅度经高斯加权 |
| 归一化 | L2 → clip(0.2) → L2 | 增强光照不变性 |

---

### 3. 特征匹配 (`matching.py`)

| 步骤 | 实现 |
|------|------|
| 距离度量 | L2 欧氏距离 `_pairwise_l2`（向量化实现 `|a|² + |b|² - 2abᵀ`） |
| 最近邻搜索 | 暴力搜索 + argpartition |
| Lowe ratio test | `best_dist < ratio * second_best_dist`（默认 `ratio=0.75`） |
| 交叉验证 | 双向匹配 + 一致性检验 `cross_check` |

---

### 4. 对极几何 (`geometry.py`)

#### 归一化 8 点法 (`compute_fundamental_8pt`)

1. **Hartley 归一化**：平移质心到原点，缩放至平均距离 √2
2. **构造 A 矩阵**：`A_i = [x₂x₁, x₂y₁, x₂, y₂x₁, y₂y₁, y₂, x₁, y₁, 1]`
3. **SVD 求解**：取 V 最后一行 → 3×3 F
4. **秩-2 约束**：SVD 后强制最小奇异值为 0
5. **去归一化**：`F = T₂ᵀ · F_norm · T₁`

#### RANSAC 鲁棒估计 (`ransac_fundamental`)

- 每次采样 8 对匹配点
- 误差度量：**Sampson 距离**（一阶几何误差近似）
- 自适应迭代次数：`n_iters = ⌈log(1-p) / log(1-w⁸)⌉`
- 最终用所有内点重新估计 F

#### 本质矩阵 (`compute_essential`)

- `E = K₂ᵀ · F · K₁`
- SVD 后强制奇异值为 `[1, 1, 0]`

---

### 5. 位姿恢复 (`pose.py`)

#### E 矩阵分解 (`decompose_essential`)

```
E = U · diag(1,1,0) · Vᵀ
```

4 组候选位姿：
| 候选 | R | t |
|------|---|---|
| 0 | `U · W · Vᵀ` | `+U[:3, 2]` |
| 1 | `U · W · Vᵀ` | `-U[:3, 2]` |
| 2 | `U · Wᵀ · Vᵀ` | `+U[:3, 2]` |
| 3 | `U · Wᵀ · Vᵀ` | `-U[:3, 2]` |

其中 `W = [[0,-1,0],[1,0,0],[0,0,1]]`

#### 正深度检验 (`select_pose`)

对每组候选 `(R, t)`，三角化所有匹配点后检查 **两个相机前方深度为正** 的点数，选取最多的那组。

---

### 6. 三角化 (`triangulation.py`)

#### DLT 线性三角化 (`triangulate_dlt`)

对每对匹配点 `(x₁, x₂)` 和投影矩阵 `(P₁, P₂)`：

```
A = [x₁[0]·P₁[2] - P₁[0]
     x₁[1]·P₁[2] - P₁[1]
     x₂[0]·P₂[2] - P₂[0]
     x₂[1]·P₂[2] - P₂[1]]
```

SVD 求解 `A · X = 0`，取最小奇异值对应的右奇异向量，齐次归一化得 3D 坐标。

#### 外点过滤 (`filter_by_error`)

- 计算重投影误差：`||proj(P, X) - x_obs||`
- 剔除误差 > `max_error` (默认 4 像素) 的点

---

### 7. 增量式 SfM (`sfm.py`)

#### 两视图初始化 (`_two_view_initialize`)

1. RANSAC + 8 点法估计基础矩阵 F
2. `E = K₂ᵀFK₁` → SVD 精化 → 4 候选 (R, t)
3. 正深度检验选取最优位姿
4. DLT 三角化初始 3D 点集
5. 重投影误差过滤

#### 增量注册 (`incremental_sfm`)

对每张新图像：

| 步骤 | 方法 |
|------|------|
| 2D-3D 对应 | 通过与已注册图像的 Track 匹配获取 |
| PnP 位姿估计 | **DLT PnP** (`_solve_pnp_dlt`)：12 参数投影矩阵 P |
| R,t 分解 | 从 P 中 QR 分解提取 `R, t = decompose(P, K)` |
| RANSAC 鲁棒 | 采样 6 对匹配，重投影误差阈值 4px |
| 新点三角化 | 用当前相机和上一个已注册相机 DLT 三角化 |
| Track 管理 | 维护每个 3D 点的多视图观测关系 |

---

### 8. 光束法平差 (`bundle_adjustment.py`)

#### 参数化

| 参数 | 表示 | 自由度 |
|------|------|--------|
| 相机旋转 | Rodrigues 轴角向量 `r ∈ ℝ³` | 3 |
| 相机平移 | 平移向量 `t ∈ ℝ³` | 3 |
| 3D 点 | 欧氏坐标 `X ∈ ℝ³` | 3 |

总变量数：`6·M + 3·N`（M 相机数，N 点数）

#### 优化方法

| 组件 | 实现 |
|------|------|
| **算法** | Levenberg-Marquardt |
| **雅可比** | 数值微分（前向差分，ε=1e⁻⁶），6 维相机 + 3 维空间点 |
| **残差** | 重投影误差：`r = x_obs − proj(K, rvec, t, X)` |
| **正规方程** | `(JᵀJ + λI) · Δx = −Jᵀr` |
| **Schur 补** | 利用相机-点结构消元，先解相机参数，再回代点参数 |
| **鲁棒核** | Huber loss：残差 > δ 时权重降为 √(δ/|r|) |
| **λ 更新** | 误差下降 → λ/2；误差上升 → 2λ |

#### Schur 补结构

正规方程分块：

```
[U    W]   [Δc]   [g_c]
[Wᵀ   V] × [Δp] = [g_p]
```

Schur 消元（V 分块对角可高效求逆）：

```
(U − WV⁻¹Wᵀ) · Δc = g_c − WV⁻¹g_p
V · Δp = g_p − WᵀΔc
```

---

### 9. 可视化与导出 (`utils.py`)

#### PLY 导出 (`export_ply`)
- ASCII 格式
- 包含顶点坐标 `(x, y, z)` 和颜色 `(r, g, b)`
- 可选法向量支持
- 可被 MeshLab / CloudCompare 打开

#### Open3D 可视化 (`visualize_open3d`)
- 稀疏点云渲染
- 相机位姿显示为彩色棱锥（Frustum）
- 每条相机射线用不同颜色标识

---

## 数学符号对照表

| 符号 | 含义 |
|------|------|
| `K` | 3×3 相机内参矩阵 |
| `F` | 3×3 基础矩阵 |
| `E` | 3×3 本质矩阵 |
| `R` | 3×3 旋转矩阵 `R ∈ SO(3)` |
| `t` | 3×1 平移向量 |
| `P` | 3×4 投影矩阵 `P = K[R|t]` |
| `X` | 3D 空间点 |
| `x` | 2D 图像点（齐次坐标） |
| `[t]_×` | 平移向量的反对称矩阵 |

---

## 已知局限

1. **特征提取速度**：DoG 极值检测 + SIFT-like 描述子为纯 Python 实现，速度远慢于 C++ 版本
2. **标定**：当前需要外部提供内参 JSON 文件，不包含棋盘格自动标定
3. **全局 SfM**：当前仅支持增量式（Incremental），不支持全局式
4. **稠密重建**：仅输出稀疏点云，无 MVS 稠密化步骤
5. **纯旋转/平面场景**：8 点法在纯旋转或平面场景下退化为多解

---

## License

MIT
