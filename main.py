import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.draw import polygon
import matplotlib.animation as animation
from fbp_baseline import AngleBinnedFBP

# 导入你写好的仿真和重建模块
from simulation import simulate_photon_data_gpu
from photon_tomo_gpu import run_solver

def create_complex_debris_target(nx, ny):
    """
    创建一个非凸、带尖锐棱角和细线条的复杂多边形空间碎片靶标
    """
    print(">>> 正在生成复杂非凸多边形碎片靶标 f(x,y)...")
    f_image = np.zeros((ny, nx), dtype=np.float32)

    # 1. 绘制带有破损帆板和天线基底的非凸多边形本体
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

    # 2. 内部遮挡结构 (挖空)
    inner_vertices = np.array([
        [-0.05, -0.1], [0.05, -0.1], [0.05, 0.1],
        [0.3, 0.05], [0.3, -0.05], [0.05, -0.05],
        [0.05, -0.3], [-0.05, -0.3], [-0.05, -0.1]
    ])
    pixel_inner_vertices = inner_vertices * scale_factor * 0.8 + np.array([nx / 2, ny / 2])
    rr_in, cc_in = polygon(pixel_inner_vertices[:, 1], pixel_inner_vertices[:, 0], shape=(ny, nx))
    f_image[rr_in, cc_in] = 0.0

    # 3. 极细线条结构 (单像素残余天线)
    line_thickness = 1
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n==========================================")
    print(f">>> MUPT 复杂非凸目标恢复闭环验证启动")
    print(f">>> 当前计算设备: {device}")
    print("==========================================\n")


    nx_target = 512
    ny_target = 512
    #f_image_true = create_complex_debris_target(nx_target, ny_target)
    f_image_true = create_two_circles_target(nx_target, ny_target)
    # 保存千万级散点数据的文件名 (建议根据参数命名)
    save_filename = "mupt_sim_data_2_200000_1.57_circle.npy"
    if save_filename.endswith('.npz'):
        photons_data = np.load(save_filename)['photons']
    else:
        photons_data = np.load(save_filename)
    print(f">>> [成功] 读取了 {len(photons_data)} 个光子事件。")

    d_theta = 1
    fbp_solver = AngleBinnedFBP(fov= 5 , nx=nx_target, d_theta_deg=d_theta)
    fbp_img = fbp_solver.reconstruct(photons_data)

    # --- 3. MUPT 逆向求解器 ---
    print("\n>>> 开始进行 MUPT 图像层析重建...")
    # 运行重构算法，建议在复杂目标下适度增加 epoch
    reconstructed_img, bounds, image_history = run_solver(
        photons_data,
        nx=nx_target,
        ny=nx_target,
        batch_size=4096,
        epochs=100,
        lr=0.01,
        fbp_img=fbp_img,
    )

    # --- 4. 终极一击：生成收敛过程 GIF 动画 ---
    print("\n>>> 正在渲染 MUPT 重建过程动画 (GIF)，请稍候...")

    delta_r = bounds[1] - bounds[0]
    extent = [-delta_r / 2, delta_r / 2, -delta_r / 2, delta_r / 2]

    fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
    ax_anim.set_xlabel("Local X (m)")
    ax_anim.set_ylabel("Local Y (m)")

    # 取最终图像的最大值作为统一色彩阈值，防止动画过程闪烁
    vmax = np.max(reconstructed_img)

    # 初始化第一帧
    im = ax_anim.imshow(image_history[0], cmap='hot', origin='lower', extent=extent, vmin=0, vmax=vmax, animated=True)
    fig_anim.colorbar(im, ax=ax_anim, fraction=0.046, pad=0.04, label='Effective Reflectivity')


    # 动画更新函数
    def update_frame(frame_idx):
        im.set_array(image_history[frame_idx])
        ax_anim.set_title(f"MUPT Evolution - Epoch {frame_idx + 1:02d}/{len(image_history)}")
        return [im]


    # 生成动画 (interval=150 表示每帧间隔 150 毫秒)
    ani = animation.FuncAnimation(fig_anim, update_frame, frames=len(image_history), interval=150, blit=True)

    # 保存为 GIF (使用 pillow writer，跨平台且无需安装额外工具)
    gif_filename = "mupt_reconstruction_evolution.gif"
    ani.save(gif_filename, writer='pillow', fps=8)
    print(f">>> [成功] 收敛过程动画已保存至本地: {gif_filename}")
    plt.close(fig_anim)  # 关闭动画的独立绘图窗口

    print("\n>>> 开始执行 FBP Baseline 扫描...")


    # --- 5. 最终静态结果对比展示 ---

    # 【强力推荐】：把图像最外围一圈 3 个像素清零，彻底杀掉 Radon 边界伪影！
    crop_margin = 3
    reconstructed_img[:crop_margin, :] = 0
    reconstructed_img[-crop_margin:, :] = 0
    reconstructed_img[:, :crop_margin] = 0
    reconstructed_img[:, -crop_margin:] = 0

    fbp_img[:crop_margin, :] = 0
    fbp_img[-crop_margin:, :] = 0
    fbp_img[:, :crop_margin] = 0
    fbp_img[:, -crop_margin:] = 0

    plt.figure(figsize=(14, 6))

    # 图 1: Ground Truth
    plt.subplot(1, 3, 1)
    plt.imshow(f_image_true, cmap='gray', origin='lower', extent=extent)
    plt.title("Ground Truth f(x,y)")

    # 图 2: MUPT 重建结果
    plt.subplot(1, 3, 2)
    # 取 99.9% 分位数作为最亮白色的天花板
    v_max_mupt = np.percentile(reconstructed_img, 99.9)
    # 核心修复：直接使用 vmin=0, vmax=v_max_mupt 让 matplotlib 强行截断！
    plt.imshow(reconstructed_img, cmap='hot', origin='lower', extent=extent, vmin=0, vmax=v_max_mupt)
    plt.title(f'Final MUPT Reconstruction\nFOV: {delta_r:.2f}m')

    # 图 3: FBP 重建结果
    plt.subplot(1, 3, 3)
    # 同样取 99.9% 截断 FBP 的星芒极值
    v_max_fbp = np.percentile(fbp_img, 99.9)
    # FBP 也使用强制截断显示
    plt.imshow(fbp_img, cmap='hot', origin='lower', extent=extent, vmin=0, vmax=v_max_fbp)
    plt.title(f'FBP Reconstruction\ntheta: {d_theta:.2f}°')

    plt.tight_layout()
    plt.show()

    # 打印真实物理绝对值，用于你在论文中分析动态范围
    print(f"\n[Dynamic Range Stats]")
    print(f"MUPT Max Value: {reconstructed_img.max():.6f} (99.9% Threshold: {v_max_mupt:.6f})")
    print(f"FBP Max Value:  {fbp_img.max():.6f} (99.9% Threshold: {v_max_fbp:.6f})")