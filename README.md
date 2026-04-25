# MUPT: 单光子直接概率层析成像新范式

**Motion-targeted Unconventional Photon Tomography**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目简介

本项目提出了一种面向运动空间碎片的**单光子直接概率层析成像新范式 (MUPT)**，通过隐式神经表示 (INR) 直接从稀疏光子散点数据重建目标图像，无需传统Radon变换的角度分箱预处理，从根本上解决了极低光子数下的成像难题。

### 核心创新

- **直接概率建模**：跳过传统弦图构建，直接对光子探测事件进行最大似然估计
- **退火位置编码**：从低频到高频逐步解锁，有效抑制散粒噪声伪影
- **FBP热启动**：利用传统方法初始化低频拓扑，加速收敛
- **GPU流式计算**：支持千万级光子数据的高效处理

## 项目结构

```
├── main.py              # 主程序入口
├── simulation.py        # GPU加速单光子物理仿真器
├── photon_tomo_gpu.py   # MUPT隐式神经表示求解器
├── fbp_baseline.py      # 传统FBP重建算法 (对比基线)
├── draw_satellite.py    # 靶标图像生成工具
└── requirements.txt     # 依赖列表
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/CuteCuteLin/-photon-tomography.git
cd -photon-tomography

# 安装依赖
pip install -r requirements.txt
```

### 依赖项

- Python >= 3.8
- PyTorch >= 2.0 (推荐GPU版本)
- NumPy >= 1.24
- Matplotlib >= 3.7
- scikit-image >= 0.21

## 快速开始

### 1. 生成仿真数据

```python
from simulation import simulate_photon_data_gpu, create_two_circles_target

# 创建靶标图像
f_image = create_two_circles_target(512, 512)

# 运行仿真
photons = simulate_photon_data_gpu(
    f_image=f_image,
    fov=5.0,           # 视场大小 (米)
    omega=0.2,         # 角速度 (rad/s)
    prf=20000,         # 脉冲重复频率 (Hz)
    time_total=15.7,   # 总观测时间 (秒)
    bin_width=32e-12,  # TDC时间精度 (秒)
    alpha=0.1,         # 光子产额系数
    eta=0.00001        # 暗计数率
)
```

### 2. 运行重建

```python
from photon_tomo_gpu import run_solver
from fbp_baseline import AngleBinnedFBP

# FBP基线重建
fbp_solver = AngleBinnedFBP(fov=5, nx=512, d_theta_deg=1)
fbp_img = fbp_solver.reconstruct(photons)

# MUPT重建
reconstructed_img, bounds, history = run_solver(
    photons,
    nx=512, ny=512,
    epochs=100,
    lr=0.01,
    fbp_img=fbp_img  # 可选：FBP热启动
)
```

## 方法原理

### 问题的起源

传统光子层析成像依赖滤波反投影（FBP）算法，需要将离散光子散点按照角度间隔进行"机械累计"，强制生成二维直方图（正弦图）。这种处理方式在单光子体制下存在致命缺陷：

1. **信息利用率极低**：机械网格划分抹平了光子到达时间中蕴含的高频空间信息
2. **异构采样引入畸变**：不同角度的光子数极度不均衡，产生条纹伪影
3. **角度失配导致运动模糊**：将连续变化角度的光子强行归结为单一"中心角度"
4. **极低光子数下数学失效**：逆Radon变换在极度稀疏的矩阵上无法收敛

### 直接光子概率层析

为从根本上消除机械累计的弊端，本项目建立从**连续散点直接映射到二维图像**的统计估计模型。

#### 正向模型

将单光子事件视为**非齐次泊松点过程（Inhomogeneous Poisson Point Process）**。探测器在角度 $\theta$、距离 $r$ 处观测到的期望光子强度为：

$$\hat{n}(\theta, r) = \mathcal{R}[f](\theta, r) + \eta$$

其中 $\mathcal{R}$ 为 Radon 变换算子，$\eta$ 为暗计数背景噪声。

#### 隐式神经表示 (INR)

将目标反射率场 $f(x,y)$ 建模为连续的坐标映射函数：

$$f(x,y) = \text{Softplus}(\text{MLP}(\gamma(x,y)))$$

采用**位置编码**克服神经网络的光谱偏置：

$$\gamma(\mathbf{x}) = [\mathbf{x}, \sin(2^k\pi\mathbf{x}), \cos(2^k\pi\mathbf{x})]_{k=0}^{L-1}$$

Softplus 激活函数从底层保证反射率的物理非负性约束 $f(x,y) \geq 0$。

#### 正向投影

期望光子强度为目标反射场与激光空间包络的 Radon 线积分：

$$\hat{n}_i = \int\int H(x,y;\theta_i,r_i) f(x,y) \, dx\,dy + \eta$$

其中系统响应函数 $H$ 为高斯型投影核。

#### 损失函数

基于泊松点过程的联合损失函数：

$$\mathcal{L} = -\sum_{i=1}^{N} \log \hat{n}_i + \lambda \int\int f(x,y) \, dx\,dy$$

- **第一项（数据保真项）**：负对数似然，强迫反射率场精确解释观测到的 $N$ 个光子散点
- **第二项（稀疏惩罚项）**：L1 形式的连续域积分，抑制空旷背景的伪影生成

#### 隐式正则化

基于 INR 的优化过程具有强大的**隐式正则化（Implicit Regularization）**效应。MLP 的网络拓扑结构和位置编码共同作用，使梯度下降自然避开高频噪声的局部极小值，优先向分段平滑且具备连续几何边缘的解空间收敛。这种"网络架构本身作为物理先验"的机理，免除了传统 TV 正则化的繁琐调参。

#### FBP热启动

利用传统滤波反投影结果初始化 INR 的低频分量，提供合理的拓扑先验，加速收敛并避免陷入局部最优。

## 联系方式

如有问题或建议，请提交 Issue 或联系作者：guoyulin@nudt.edu.cn
