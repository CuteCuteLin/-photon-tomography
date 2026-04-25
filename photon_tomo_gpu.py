import torch
import torch.nn as nn
import numpy as np
import time
import sys
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR


# ==========================================
# 1. 隐式神经表示 (INR) 核心模块
# ==========================================
class AnnealedPositionalEncoding(nn.Module):
    """
    带频率退火的二维位置编码 (Coarse-to-Fine Frequency Annealing)
    彻底解决极低光子数下的高频等间距散点伪影！
    """

    def __init__(self, num_frequencies=4):
        super().__init__()
        self.num_frequencies = num_frequencies
        # 动态控制高频解锁的进度，初始为 0
        self.register_buffer('progress', torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        freqs = 2 ** torch.arange(self.num_frequencies, dtype=torch.float32, device=x.device) * np.pi
        encoded = (x.unsqueeze(-1) * freqs).view(x.shape[0], -1)

        # 核心数学：计算平滑截断权重 (BARF-style)
        # alpha 从 0 逐渐增大到 num_frequencies
        alpha = self.progress * self.num_frequencies
        k = torch.arange(self.num_frequencies, dtype=torch.float32, device=x.device)

        # 权重计算：利用 cosine 实现平滑过渡 (0 -> 1)
        weight = (1.0 - torch.cos(np.pi * torch.clamp(alpha - k, min=0.0, max=1.0))) / 2.0

        # 因为 x 和 y 两个维度被展平了，权重需要 repeat_interleave
        weight = weight.repeat_interleave(2)

        # 用动态权重压制高频
        sin_enc = torch.sin(encoded) * weight
        cos_enc = torch.cos(encoded) * weight

        return torch.cat([x, sin_enc, cos_enc], dim=-1)


class MUPT_INRSolver(nn.Module):
    def __init__(self, nx, ny, fov):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.fov = fov

        # ==========================================
        # 1. 恢复高频感知
        # ==========================================
        # 频率恢复到 6，让网络重新获得描绘极度锐利边缘的能力
        self.pos_enc = AnnealedPositionalEncoding(num_frequencies=6)
        in_dim = 2 + 2 * 2 * 6

        hidden_dim = 256
        # 纯净的 MLP，使用 GELU 保持丝滑，末端不加任何激活函数！
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # ==========================================
        # 2. 物理常数与坐标系
        # ==========================================
        self.sigma_proj = (self.fov / self.nx) * 1.0
        self.register_buffer('fixed_eta', torch.tensor([0.0001], dtype=torch.float32))

        x = torch.linspace(-self.fov / 2, self.fov / 2, nx)
        y = torch.linspace(-self.fov / 2, self.fov / 2, ny)
        X, Y = torch.meshgrid(x, y, indexing='xy')

        self.register_buffer('X_flat', X.reshape(-1))
        self.register_buffer('Y_flat', Y.reshape(-1))
        self.coords = torch.stack([self.X_flat, self.Y_flat], dim=-1)

    def forward(self, thetas, rs):
        B = thetas.shape[0]
        thetas = thetas.view(B, 1)
        rs = rs.view(B, 1)
        device = thetas.device

        # ==========================================
        # 3. 极简前向传播 (无梯度追踪，极速运行)
        # ==========================================
        coords_batch = self.coords.to(device)
        encoded_coords = self.pos_enc(coords_batch)

        f_pred = self.net(encoded_coords).squeeze()
        # 用 softplus 保证密度为正
        f_positive = torch.nn.functional.softplus(f_pred) + 1e-6

        # ==========================================
        # 4. Radon 投影
        # ==========================================
        proj_dist = self.X_flat * torch.cos(thetas) + self.Y_flat * torch.sin(thetas) - rs
        H_batch = torch.exp(-(proj_dist ** 2) / (2 * self.sigma_proj ** 2))

        expected_photons = torch.matmul(H_batch, f_positive) + self.fixed_eta + 1e-9

        # ==========================================
        # 5. 纯粹的 Loss 博弈
        # ==========================================
        # 绝对主导：拟合光子数据
        nll_loss = -torch.mean(torch.log(expected_photons))

        # 唯一的防线：极弱的 L1 稀疏惩罚，用来吸走背景的灰雾
        sparsity_loss = torch.mean(f_positive)

        # 0.05 足以维持宇宙背景的黑暗，网络可以尽情在光子处画高光
        total_loss = nll_loss + 0.05 * sparsity_loss

        return total_loss

    def get_image(self):
        # ==========================================
        # 【核心修复】：自动获取模型当前所在的设备 (GPU)，
        # 并把坐标网格推送到该设备上！
        # ==========================================
        device = next(self.parameters()).device
        coords_batch = self.coords.to(device)

        encoded_coords = self.pos_enc(coords_batch)
        f_pred = self.net(encoded_coords).squeeze()

        # 所见即所得，直接输出发光密度
        f_img = torch.nn.functional.softplus(f_pred)
        return f_img

# ==========================================
# 2. 调度主程序 (带自动显存管控与动态进度条)
# ==========================================
class PhotonDataset(Dataset):
    def __init__(self, data): self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self): return self.data.shape[0]

    def __getitem__(self, idx): return self.data[idx, 0], self.data[idx, 1]

def run_solver(photons_np, target_fov=5.0, nx=256, ny=256, batch_size="auto", epochs=60, lr=0.005,fbp_img = None):
    #执行 MUPT 算法主流程
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MUPT_INRSolver(nx, ny, fov=target_fov).to(device)
    print(f"\n>>> 启动 MUPT-INR GPU 重建求解器 (Device: {device})")
    print(f"  > 重构目标 FOV: {target_fov} m | 图像分辨率: {nx}x{ny}")

    # ==========================================
    # 【神来之笔：低频锁死的 FBP 热启动】
    # ==========================================
    if fbp_img is not None:
        print("\n>>> [阶段一] 开始低频 FBP 热启动 (只吸收拓扑，免疫条纹)...")
        fbp_flat = np.clip(fbp_img.flatten(), a_min=1e-6, a_max=None)
        target_tensor = torch.tensor(fbp_flat, dtype=torch.float32, device=device)

        # 正常的学习率即可，预训练次数骤降到 30~50 次！
        pre_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        pre_epochs = 40

        # 【核心操作：强制锁死在高频退火的最早期】
        # progress=0.15 意味着只激活最低频的神经元，网络根本画不出高频的细小条纹！
        model.pos_enc.progress.fill_(0.1)

        for p_epoch in range(pre_epochs):
            pre_optimizer.zero_grad()
            pred_img = model.get_image().flatten()

            # 拟合 FBP
            mse_loss = torch.mean((pred_img - target_tensor) ** 2)
            mse_loss.backward()
            pre_optimizer.step()

            if (p_epoch + 1) % 10 == 0:
                sys.stdout.write(f"\r    [低频预训练 {p_epoch + 1}/{pre_epochs}] MSE Loss: {mse_loss.item():.6f}")
                sys.stdout.flush()

        print("\n>>> 低频拓扑骨架提取完毕！准备进入全频段光子雕刻...\n")


    # ==========================================
    # 核心：动态显存管控 (计算最安全的 photon_batch_size)
    # ==========================================
    N_pixels = nx * ny
    if batch_size == "auto":
        # 我们限制 H_batch 最大占用 1.5GB 显存 (1.5e9 bytes)
        # 每个 float32 占 4 字节，计算公式: B * N_pixels * 4 = 1.5e9
        safe_b = int(1.5e9 / (4 * N_pixels))
        # 钳制在合理范围内
        batch_size = max(128, min(10000, safe_b))

    print(f"  > 自动安全分批大小 (Photon Batch Size): {batch_size}")

    dataset = PhotonDataset(photons_np)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

    image_history = []
    total_batches = len(dataloader)

    print("-" * 65)
    model.train()
    # 设定在多少个 epoch 内完成从粗到细的过渡 (比如前 50% 的时间)
    anneal_epochs = epochs * 0.8
    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()
        # 【新增】：动态更新频率解锁进度！

        # progress 严格钳制在 [0, 1] 之间
        current_progress = min(1.0, 0.1 + epoch / anneal_epochs)
        model.pos_enc.progress.fill_(current_progress)

        for i, (batch_thetas, batch_rs) in enumerate(dataloader):
            batch_thetas = batch_thetas.to(device)
            batch_rs = batch_rs.to(device)

            optimizer.zero_grad()
            loss = model(batch_thetas, batch_rs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # ==========================================
            # 炫酷的实时进度条 (ETA)
            # ==========================================
            percent = (i + 1) / total_batches
            elapsed = time.time() - start_time
            eta = (elapsed / percent) - elapsed if percent > 0 else 0

            bar_len = 20
            filled = int(bar_len * percent)
            bar = '=' * filled + '-' * (bar_len - filled)

            # 使用 \r 实现单行刷新
            sys.stdout.write(f"\r    [Epoch {epoch + 1:02d}] 进度: [{bar}] {percent * 100:5.1f}% | ETA: {eta:5.1f}s ")
            sys.stdout.flush()

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        avg_loss = epoch_loss / total_batches
        elapsed = time.time() - start_time

        # 覆盖掉子进度条，打印该 Epoch 最终结果，并换行
        sys.stdout.write(
            f"\rEpoch {epoch + 1:02d}/{epochs} | Loss: {avg_loss:10.4f} | LR: {current_lr:.5f} | Time: {elapsed:5.2f}s        \n")

        with torch.no_grad():
            current_img = model.get_image().cpu().numpy().reshape((ny, nx)).copy()
            image_history.append(current_img)

    with torch.no_grad():
        final_image = model.get_image().cpu().numpy().reshape((ny, nx))

    bounds = [-target_fov / 2, target_fov / 2]
    return final_image, bounds, image_history

