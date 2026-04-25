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

### 物理模型

系统模拟旋转扫描单光子LiDAR对空间碎片进行观测。探测器绕目标旋转，每个光子探测事件记录为 $(\theta_i, r_i)$：
- $\theta_i$: 第 $i$ 个光子被探测时的扫描角度
- $r_i$: 该光子的相对距离测量值

### 正向模型

对于给定的目标反射率场 $f(x,y)$，单个光子事件 $(\theta, r)$ 的期望探测计数为：

$$\hat{n}(\theta, r) = \int\int H(x,y;\theta,r) f(x,y) \, dx\,dy + \eta$$

其中 $H(x,y;\theta,r)$ 为系统响应函数（高斯型投影核）：

$$H(x,y;\theta,r) = \exp\left(-\frac{(x\cos\theta + y\sin\theta - r)^2}{2\sigma^2}\right)$$

$\eta$ 为暗计数背景噪声。

### 隐式神经表示 (INR)

目标反射率场 $f(x,y)$ 由多层感知机参数化：

$$f(x,y) = \text{Softplus}(\text{MLP}(\gamma(x,y)))$$

其中 $\gamma(\cdot)$ 为**退火位置编码**：

$$\gamma(\mathbf{x}) = [\mathbf{x}, \sin(2^k\pi\mathbf{x}), \cos(2^k\pi\mathbf{x})]_{k=0}^{L-1} \cdot w_k(\alpha)$$

权重 $w_k(\alpha)$ 随训练进度 $\alpha$ 从0平滑过渡到1，实现从低频到高频的渐进式解锁，有效抑制散粒噪声引入的高频伪影。

### 损失函数

采用负对数似然损失配合稀疏正则化：

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log \hat{n}(\theta_i, r_i) + \lambda \cdot \text{mean}(f)$$

- 第一项：负对数似然损失，驱动模型拟合观测到的光子分布
- 第二项：L1稀疏正则化，抑制背景区域的虚假响应

### FBP热启动

利用传统滤波反投影 (FBP) 结果初始化INR的低频分量，提供合理的拓扑先验，显著加速收敛并避免陷入局部最优。

## 联系方式

如有问题或建议，请提交 Issue 或联系作者：guoyulin@nudt.edu.cn
