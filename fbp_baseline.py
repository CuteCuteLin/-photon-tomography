import numpy as np


class AngleBinnedFBP:
    """
    基于角度累积 (Angle Binning) 的传统滤波反投影 (FBP) 算法。
    专为极弱单光子 LiDAR 散点数据设计，用于作为 MUPT 范式的 Baseline 对比。
    """

    def __init__(self, fov, nx, d_theta_deg=1.0):
        """
        初始化 FBP 重建器
        :param fov: 视场大小 (物理绝对宽度，单位：米)
        :param nx: 图像重建的分辨率 (nx * nx)
        :param d_theta_deg: 核心参数！角度累积窗口 (单位：度)
        """
        self.fov = fov
        self.nx = nx
        self.d_theta_rad = np.radians(d_theta_deg)
        self.d_theta_deg = d_theta_deg

        # 计算对角线视场，确保边缘目标在旋转时不会出界
        self.diag_fov = self.fov * np.sqrt(2)

        # 匹配图像分辨率的波形分辨率 (dr)
        self.pixel_size = self.fov / self.nx
        self.num_r_bins = int(np.ceil(self.diag_fov / self.pixel_size))

        # 定义探测器的物理网格
        self.r_bins = np.linspace(-self.diag_fov / 2, self.diag_fov / 2, self.num_r_bins)
        self.dr = self.r_bins[1] - self.r_bins[0]

        # 定义图像的物理坐标网格
        x = np.linspace(-self.fov / 2, self.fov / 2, self.nx)
        y = np.linspace(-self.fov / 2, self.fov / 2, self.nx)
        self.X, self.Y = np.meshgrid(x, y)

    def _build_sinogram(self, photons_data):
        """阶段 1：将无结构的离散光子散点，强行填入直方图 (Angle Binning)"""
        thetas_rad = photons_data[:, 0]
        rs = photons_data[:, 1]

        # 确定角度的边界 (0 到 2pi 或实际的最大最小角度)
        min_theta, max_theta = np.min(thetas_rad), np.max(thetas_rad)

        # 划分角度 bins
        theta_bins = np.arange(min_theta, max_theta + self.d_theta_rad, self.d_theta_rad)
        num_theta_bins = len(theta_bins) - 1

        sinogram = np.zeros((self.num_r_bins, num_theta_bins))
        theta_centers = np.zeros(num_theta_bins)

        # 二维直方图统计 (这正是传统方法丢失时空分辨率的罪魁祸首)
        for i in range(num_theta_bins):
            t_start = theta_bins[i]
            t_end = theta_bins[i + 1]
            theta_centers[i] = (t_start + t_end) / 2.0

            # 提取落在这个角度窗口内的所有光子
            mask = (thetas_rad >= t_start) & (thetas_rad < t_end)
            rs_in_bin = rs[mask]

            # 在距离维度上做直方图统计
            counts, _ = np.histogram(rs_in_bin, bins=self.num_r_bins, range=(-self.diag_fov / 2, self.diag_fov / 2))
            sinogram[:, i] = counts

        return sinogram, theta_centers

    def _ramp_filter(self, sinogram):
        """阶段 2：频域 Ramp 滤波 (高频放大器，引发散粒噪声灾难的元凶)"""
        num_r, num_theta = sinogram.shape

        # FFT 补零，防止循环卷积伪影
        n_fft = int(2 ** np.ceil(np.log2(2 * num_r)))

        # 构建 Ramp 滤波器 |omega|
        freqs = np.fft.fftfreq(n_fft, d=self.dr)
        ramp = np.abs(freqs)

        # 频域滤波
        sino_fft = np.fft.fft(sinogram, n=n_fft, axis=0)
        filtered_fft = sino_fft * ramp[:, np.newaxis]

        # 逆傅里叶变换，并截断回原尺寸
        filtered_sino = np.fft.ifft(filtered_fft, axis=0).real
        return filtered_sino[:num_r, :]

    def _back_project(self, filtered_sino, theta_centers):
        """阶段 3：空间反投影"""
        img = np.zeros((self.nx, self.nx))
        num_thetas = len(theta_centers)

        for i, theta in enumerate(theta_centers):
            # 计算图像上每个像素到当前角度射线的垂直投影距离
            r_proj = self.X * np.cos(theta) + self.Y * np.sin(theta)

            # 从滤波后的波形中插值获取能量
            # 注意：np.interp 要求横坐标是递增的
            img += np.interp(r_proj, self.r_bins, filtered_sino[:, i], left=0, right=0)

        # 乘以角度步长完成积分
        img *= (np.pi / num_thetas)

        # 将负值截断 (FBP 的固有振铃效应会产生负值)
        img[img < 0] = 0
        return img

    def reconstruct(self, photons_data):
        """
        执行完整的 FBP 重建流水线
        """
        print(f"--- FBP Baseline (d_theta = {self.d_theta_deg} deg) ---")

        # 1. Angle Binning
        sinogram, theta_centers = self._build_sinogram(photons_data)
        total_photons = np.sum(sinogram)
        print(f"  > 构建弦图: {sinogram.shape[1]} 个角度 bins, 包含 {total_photons:.0f} 个光子")

        # 2. 滤波
        filtered_sino = self._ramp_filter(sinogram)

        # 3. 反投影
        print("  > 正在进行反投影...")
        recon_img = self._back_project(filtered_sino, theta_centers)

        return recon_img