import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
import torch
import torch.nn.functional as F
import time
import sys


def simulate_photon_data_gpu(f_image, fov, omega, prf, time_total, bin_width, alpha=1.0, eta=0.01, batch_size="auto"):
    """
    基于流式批处理 (Streaming Batch) 的极速单光子物理仿真器
    彻底解决超大分辨率下的 CUDA Out of Memory 问题，并自带 ETA 进度条。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> 启动极速物理仿真 (计算设备: {device})...")
    start_time = time.time()

    # ==========================================
    # 阶段 1：计算物理常数与硬件 TDC 参数
    # ==========================================
    c = 3e8
    dr_tdc = bin_width * c / 2.0
    diag_fov = fov * np.sqrt(2)

    tdc_bins = int(np.ceil(diag_fov / dr_tdc))
    print(f"  > TDC 时间精度: {bin_width * 1e12:.1f} ps")
    print(f"  > TDC 物理距离分辨率 (dr): {dr_tdc * 1000:.2f} mm")
    print(f"  > 真实探测波形长度 (TDC Bins): {tdc_bins} 个")

    num_pulses = int(time_total * prf)
    t = torch.arange(num_pulses, device=device, dtype=torch.float32) / prf
    thetas_rad = omega * t

    N_orig = f_image.shape[0]
    N_pad = int(np.ceil(N_orig * np.sqrt(2)))
    pad_val = (N_pad - N_orig) // 2

    f_tensor = torch.tensor(f_image, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    f_padded = F.pad(f_tensor, (pad_val, N_pad - N_orig - pad_val, pad_val, N_pad - N_orig - pad_val), mode='constant',
                     value=0)

    # ==========================================
    # 阶段 1.5：动态显存安全管控与全局最大值预估
    # ==========================================
    if batch_size == "auto":
        # 预估 F.affine_grid 产生的显存 (限制每个 batch 约占用 1.5GB 显存)
        safe_batch = int(2e9 / (12 * N_pad * N_pad))
        batch_size = max(10, safe_batch)
    print(f"  > 自动安全分批大小 (Batch Size): {batch_size}")

    # 预计算全局最大 Radon 积分 (用于归一化)。只需抽样几个角度即可精准获取
    print("  > 正在预计算目标的全局反射截面积极限...")
    with torch.no_grad():
        test_thetas = torch.linspace(0, np.pi, 36, device=device)
        cos_t, sin_t = torch.cos(test_thetas), torch.sin(test_thetas)
        theta_mat = torch.zeros((36, 2, 3), device=device)
        theta_mat[:, 0, 0], theta_mat[:, 0, 1] = cos_t, -sin_t
        theta_mat[:, 1, 0], theta_mat[:, 1, 1] = sin_t, cos_t

        grid = F.affine_grid(theta_mat, [36, 1, N_pad, N_pad], align_corners=False)
        rotated = F.grid_sample(f_padded.expand(36, -1, -1, -1), grid, align_corners=False)

        sino_test = rotated.sum(dim=2)  # [36, 1, N_pad]
        sino_tdc = F.interpolate(sino_test, size=tdc_bins, mode='linear', align_corners=True).squeeze(1)
        max_pulse_sum = sino_tdc.sum(dim=1).max().item()

    # ==========================================
    # 阶段 2：全链路流式批处理生成光子 (彻底防爆显存)
    # ==========================================
    print(f">>> 开始执行流式光子生成流水线 (共 {num_pulses} 个脉冲)...")
    all_photons = []
    r_bins_tdc = torch.linspace(-diag_fov / 2, diag_fov / 2, tdc_bins, device=device)

    for i in range(0, num_pulses, batch_size):
        end_idx = min(i + batch_size, num_pulses)
        theta_batch = thetas_rad[i:end_idx]
        B = len(theta_batch)

        with torch.no_grad():
            # 1. 坐标旋转矩阵
            cos_t, sin_t = torch.cos(theta_batch), torch.sin(theta_batch)
            theta_mat = torch.zeros((B, 2, 3), device=device)
            theta_mat[:, 0, 0], theta_mat[:, 0, 1] = cos_t, -sin_t
            theta_mat[:, 1, 0], theta_mat[:, 1, 1] = sin_t, cos_t

            # 2. 图像重采样与积分 (Radon变换)
            grid = F.affine_grid(theta_mat, [B, 1, N_pad, N_pad], align_corners=False)
            rotated_images = F.grid_sample(f_padded.expand(B, -1, -1, -1), grid, align_corners=False)
            sinogram_batch = rotated_images.sum(dim=2)  # [B, 1, N_pad]

            # 3. 硬件分辨率升采样插值
            sinogram_tdc = F.interpolate(sinogram_batch, size=tdc_bins, mode='linear', align_corners=True).squeeze(1)

            # 4. 物理截面积归一化
            sinogram_normalized = sinogram_tdc / (max_pulse_sum + 1e-9)

            # 5. 泊松散粒噪声抽样
            Lambda_matrix = alpha * sinogram_normalized + eta
            photon_counts = torch.poisson(Lambda_matrix)

            # 6. 坐标提取与抖动连续化
            hit_indices = torch.nonzero(photon_counts)
            if len(hit_indices) > 0:
                pulse_idx = hit_indices[:, 0]
                r_idx = hit_indices[:, 1]
                counts = photon_counts[pulse_idx, r_idx].long()

                pulse_idx_repeated = torch.repeat_interleave(pulse_idx, counts)
                r_idx_repeated = torch.repeat_interleave(r_idx, counts)

                sampled_thetas = theta_batch[pulse_idx_repeated]
                sampled_rs_discrete = r_bins_tdc[r_idx_repeated]

                # 亚分辨率硬件抖动
                jitter = (torch.rand_like(sampled_rs_discrete) - 0.5) * dr_tdc
                sampled_rs_continuous = sampled_rs_discrete + jitter

                # 保存这一个 Batch 收集到的光子，立即推入 CPU，释放 GPU 显存
                batch_photons = np.column_stack((sampled_thetas.cpu().numpy(), sampled_rs_continuous.cpu().numpy()))
                all_photons.append(batch_photons)

        # 动态更新进度条与 ETA
        processed = end_idx
        percent = processed / num_pulses
        elapsed = time.time() - start_time
        eta_time = (elapsed / percent) - elapsed if percent > 0 else 0

        bar_len = 30
        filled = int(bar_len * percent)
        bar = '=' * filled + '-' * (bar_len - filled)
        sys.stdout.write(
            f"\r  [{bar}] {percent * 100:.1f}% | 脉冲: {processed}/{num_pulses} | 预计剩余: {eta_time:.1f}s ")
        sys.stdout.flush()

    # 合并所有散点
    if len(all_photons) > 0:
        photons_np = np.vstack(all_photons)
    else:
        photons_np = np.empty((0, 2))

    total_time = time.time() - start_time
    print(f"\n>>> 仿真圆满完成！总耗时 {total_time:.1f} 秒，共生成了 {len(photons_np)} 个单光子事件。")
    return photons_np


def create_complex_debris_target(nx, ny):
    """复杂碎片靶标生成 (保持不变)"""
    print(">>> 正在生成复杂非凸多边形碎片靶标 f(x,y)...")
    f_image = np.zeros((ny, nx), dtype=np.float32)

    vertices = np.array([
        [-0.1, -0.6], [0.1, -0.6], [0.2, -0.2],
        [0.8, -0.2], [0.9, 0.0], [0.8, 0.2],
        [0.2, 0.2], [0.1, 0.6], [-0.1, 0.6],
        [-0.2, 0.2], [-0.8, 0.2], [-0.9, 0.0],
        [-0.8, -0.2], [-0.2, -0.2], [-0.1, -0.6]
    ])
    scale_factor = 0.4 * min(nx, ny)
    pixel_vertices = vertices * scale_factor + np.array([nx / 2, ny / 2])
    rr, cc = polygon(pixel_vertices[:, 1], pixel_vertices[:, 0], shape=(ny, nx))
    f_image[rr, cc] = 1.0

    inner_vertices = np.array([
        [-0.05, -0.1], [0.05, -0.1], [0.05, 0.1],
        [0.3, 0.05], [0.3, -0.05], [0.05, -0.05],
        [0.05, -0.3], [-0.05, -0.3], [-0.05, -0.1]
    ])
    pixel_inner_vertices = inner_vertices * scale_factor * 0.8 + np.array([nx / 2, ny / 2])
    rr_in, cc_in = polygon(pixel_inner_vertices[:, 1], pixel_inner_vertices[:, 0], shape=(ny, nx))
    f_image[rr_in, cc_in] = 0.0

    line_thickness = max(1, int(min(nx, ny) * 0.005))  # 天线宽度自适应分辨率
    f_image[int(ny / 2 - scale_factor * 0.3): int(ny / 2 - scale_factor * 0.3 + line_thickness),
    int(nx / 2 + scale_factor * 0.5): int(nx / 2 + scale_factor * 0.9)] = 0.8
    f_image[int(ny / 2 + scale_factor * 0.5): int(ny / 2 + scale_factor * 0.9),
    int(nx / 2 - scale_factor * 0.3): int(nx / 2 - scale_factor * 0.3 + line_thickness)] = 0.8

    return f_image


def create_two_circles_target(nx, ny, radius_ratio=0.08, thickness_ratio=0.1):
    """
    生成由两个空心圆环组成的靶标图像 f(x,y)

    参数:
    - nx, ny: 图像的宽度和高度 (像素)
    - radius_ratio: 圆环外半径占图像短边的比例 (默认 0.08)
    - thickness_ratio: 圆环厚度占外半径的比例 (0~1之间，默认 0.3 表示厚度是半径的30%)

    返回:
    - f_image: numpy 数组，值为 1.0 代表圆环，0.0 代表背景
    """
    print(">>> 正在生成双空心圆环靶标 f(x,y)...")
    f_image = np.zeros((ny, nx), dtype=np.float32)

    min_dim = min(nx, ny)
    outer_radius = int(min_dim * radius_ratio)

    # 计算内半径，确保厚度合理
    # 厚度 = 外半径 * thickness_ratio
    # 内半径 = 外半径 - 厚度
    thickness = max(1, int(outer_radius * thickness_ratio))
    inner_radius = outer_radius - thickness

    if inner_radius < 0:
        inner_radius = 0
        print(f"警告：圆环厚度过大，已自动调整为实心圆。")

    # 定义两个圆心的位置 (非对称分布)
    # 这里设置两个明显不对称的坐标，避免关于中心对称
    # 圆1：左下区域
    center1_x = int(nx * 0.3)
    center1_y = int(ny * 0.4)

    # 圆2：右上区域，且大小可以略有不同（如果需要，这里保持半径一致仅位置不同）
    center2_x = int(nx * 0.75)
    center2_y = int(ny * 0.65)

    circle1_center = (center1_x, center1_y)
    circle2_center = (center2_x, center2_y)

    print(f"圆1中心: {circle1_center}, 外半径: {outer_radius}, 内半径: {inner_radius}")
    print(f"圆2中心: {circle2_center}, 外半径: {outer_radius}, 内半径: {inner_radius}")

    # 生成网格坐标
    y_coords, x_coords = np.ogrid[:ny, :nx]

    # 计算到两个圆心的距离
    dist1 = np.sqrt((x_coords - circle1_center[0]) ** 2 + (y_coords - circle1_center[1]) ** 2)
    dist2 = np.sqrt((x_coords - circle2_center[0]) ** 2 + (y_coords - circle2_center[1]) ** 2)

    # 创建空心圆环掩膜：距离大于内半径 且 小于等于外半径
    mask1 = (dist1 > inner_radius) & (dist1 <= outer_radius)
    mask2 = (dist2 > inner_radius) & (dist2 <= outer_radius)

    # 将圆环区域设为 1.0
    f_image[mask1 | mask2] = 1.0

    return f_image


# ==========================================
# 验证模块
# ==========================================
if __name__ == "__main__":
    # 你现在可以随意调大分辨率了，绝不会爆显存
    N_pixels = 512
    f_image_true = create_complex_debris_target(N_pixels, N_pixels)
    #f_image_true = create_two_circles_target(N_pixels, N_pixels)
    sim_params = {
        'f_image': f_image_true,
        'fov': 5.0,
        'omega': 0.2,
        'prf': 20000,
        'time_total': 15.7,
        'bin_width': 32e-12,  # 64 ps
        'alpha': 0.1,
        'eta': 0.00001,
        'batch_size': 'auto'  # 核心：开启自动显存管控
    }

    photons_sim = simulate_photon_data_gpu(**sim_params)

    save_filename = f"mupt_sim_data_{sim_params['omega']}_{sim_params['prf']}_{sim_params['time_total']}.npy"
    np.save(save_filename, photons_sim)
    print(f"\n>>> [成功] 数据已保存至: {save_filename}")
    print(f">>> 文件大小约: {photons_sim.nbytes / (1024 * 1024):.2f} MB")

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(f_image_true, cmap='gray', extent=[-2.5, 2.5, -2.5, 2.5], origin='lower')
    plt.title(f"Ground Truth f(x,y)\nResolution: {N_pixels}x{N_pixels}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.subplot(1, 2, 2)
    plot_limit = min(5000000, len(photons_sim))
    plt.scatter(photons_sim[:plot_limit, 0], photons_sim[:plot_limit, 1], s=0.1, alpha=0.3, color='blue')
    plt.title(
        f"Simulated Photon Scatter (Showing {plot_limit}/{len(photons_sim)})\nTDC Bin Width: {sim_params['bin_width'] * 1e12:.0f}ps")
    plt.xlabel("Angle $\\theta$ (rad)")
    plt.ylabel("Relative Distance $r$ (m)")

    plt.tight_layout()
    plt.show()