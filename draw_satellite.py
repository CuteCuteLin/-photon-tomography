import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


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


# --- 测试代码 ---
if __name__ == "__main__":
    N_pixels = 512
    # 调用函数，厚度设为半径的 40%
    f_image_true = create_two_circles_target(N_pixels, N_pixels, radius_ratio=0.1, thickness_ratio=0.2)

    plt.figure(figsize=(6, 6))
    plt.imshow(f_image_true, cmap='gray', extent=[-2.5, 2.5, -2.5, 2.5], origin='lower')
    plt.title("Two Asymmetric Hollow Circles (Ring Targets)")
    plt.axis('off')
    plt.show()
