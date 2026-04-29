import numpy as np

# 生成高斯模糊的权重模板
def _gaussian_kernel_1d(sigma, radius=None):
    if radius is None:
        radius = int(np.ceil(3.0 * sigma))
    if radius < 1:
        radius = 1
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel

# 构建高斯金字塔
def _build_gaussian_pyramid(img_gray, n_octaves, n_scales, sigma0):
    if img_gray.dtype != np.float64:
        base = img_gray.astype(np.float64) / 255.0
    else:
        base = img_gray.copy()

    sigma_prev = sigma0
    pyramid = []
    for o in range(n_octaves):
        octave = []
        if o == 0:
            current = base
        else:
            prev_octave = pyramid[-1]
            current = prev_octave[-3][::2, ::2]

        sigma_current = sigma_prev
        for s in range(n_scales + 3):
            if s == 0 and o == 0:
                blur = _gaussian_blur(base, sigma0)
                octave.append(blur)
            elif s == 0:
                octave.append(current)
            else:
                sigma_eff = np.sqrt(sigma_current**2 - sigma_prev**2) if sigma_current > sigma_prev else sigma_current * 0.5
                if sigma_eff < 0.01:
                    sigma_eff = sigma_current
                blur = _gaussian_blur(current, sigma_eff)
                octave.append(blur)
            sigma_prev = sigma_current
            sigma_current *= 2.0 ** (1.0 / n_scales)

        pyramid.append(octave)

    return pyramid

# 分离卷积函数
def _convolve1d_separable(img_2d, kernel_1d, axis):
    k = kernel_1d
    radius = len(k) // 2
    h, w = img_2d.shape
    result = np.zeros_like(img_2d)
    if axis == 1:
        padded = np.pad(img_2d, ((0, 0), (radius, radius)), mode='reflect')
        for i in range(w):
            result[:, i] = np.sum(padded[:, i:i + len(k)] * k, axis=1)
    else:
        padded = np.pad(img_2d, ((radius, radius), (0, 0)), mode='reflect')
        for i in range(h):
            result[i, :] = np.sum(padded[i:i + len(k), :] * k[:, None], axis=0)
    return result

# 高斯模糊函数
def _gaussian_blur(img, sigma):
    if sigma < 0.3:
        return img.copy()
    kernel = _gaussian_kernel_1d(sigma)
    tmp = _convolve1d_separable(img, kernel, axis=1)
    return _convolve1d_separable(tmp, kernel, axis=0)

# 构建 DoG 金字塔
def _build_dog_pyramid(gaussian_pyramid):
    dog_pyramid = []
    for octave in gaussian_pyramid:
        dog_octave = []
        for s in range(len(octave) - 1):
            dog = octave[s + 1] - octave[s]
            dog_octave.append(dog)
        dog_pyramid.append(dog_octave)
    return dog_pyramid

# 检测极值点
def _detect_extrema(dog_pyramid, n_scales, contrast_threshold, edge_threshold):
    keypoints = []
    for o, dog_octave in enumerate(dog_pyramid):
        h, w = dog_octave[0].shape
        for s in range(1, n_scales):
            dog_slice = np.stack([
                dog_octave[s - 1],
                dog_octave[s],
                dog_octave[s + 1],
            ], axis=0)

            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    val = dog_slice[1, y, x]
                    if np.abs(val) < contrast_threshold:
                        continue
                    patch = dog_slice[:, y - 1:y + 2, x - 1:x + 2]
                    is_max = val >= patch.max()
                    is_min = val <= patch.min()
                    if is_max or is_min:
                        if _check_edge(dog_octave[s], x, y, edge_threshold):
                            continue
                        kp = _refine_keypoint(dog_octave, s, x, y, contrast_threshold, edge_threshold)
                        if kp is not None:
                            x_oct, y_oct, sigma_oct = kp
                            if 0 <= x_oct < w and 0 <= y_oct < h:
                                scale = 2.0 ** o
                                keypoints.append((x_oct * scale, y_oct * scale, sigma_oct * scale))
    return keypoints

# 边缘响应检查
def _check_edge(dog, x, y, edge_threshold):
    if x < 1 or y < 1 or x >= dog.shape[1] - 1 or y >= dog.shape[0] - 1:
        return True
    dx = (dog[y, x + 1] - dog[y, x - 1]) / 2.0
    dy = (dog[y + 1, x] - dog[y - 1, x]) / 2.0
    dxx = dog[y, x + 1] + dog[y, x - 1] - 2.0 * dog[y, x]
    dyy = dog[y + 1, x] + dog[y - 1, x] - 2.0 * dog[y, x]
    dxy = ((dog[y + 1, x + 1] - dog[y + 1, x - 1]) -
           (dog[y - 1, x + 1] - dog[y - 1, x - 1])) / 4.0

    tr = dxx + dyy
    det = dxx * dyy - dxy * dxy
    if det <= 0:
        return True
    curv_ratio = tr * tr / det
    r = edge_threshold
    return curv_ratio > (r + 1) * (r + 1) / r

# 关键点精细化
def _refine_keypoint(dog_octave, s, x, y, contrast_threshold, edge_threshold):
    for i in range(5):
        dog = dog_octave[s]
        if x < 1 or y < 1 or x >= dog.shape[1] - 1 or y >= dog.shape[0] - 1:
            return None

        dx = (dog[y, x + 1] - dog[y, x - 1]) / 2.0
        dy = (dog[y + 1, x] - dog[y - 1, x]) / 2.0
        ds = (dog_octave[s + 1][y, x] - dog_octave[s - 1][y, x]) / 2.0

        dxx = dog[y, x + 1] + dog[y, x - 1] - 2.0 * dog[y, x]
        dyy = dog[y + 1, x] + dog[y - 1, x] - 2.0 * dog[y, x]
        dss = dog_octave[s + 1][y, x] + dog_octave[s - 1][y, x] - 2.0 * dog[y, x]

        dxy = ((dog[y + 1, x + 1] - dog[y + 1, x - 1]) -
               (dog[y - 1, x + 1] - dog[y - 1, x - 1])) / 4.0
        dxs = ((dog_octave[s + 1][y, x + 1] - dog_octave[s + 1][y, x - 1]) -
               (dog_octave[s - 1][y, x + 1] - dog_octave[s - 1][y, x - 1])) / 4.0
        dys = ((dog_octave[s + 1][y + 1, x] - dog_octave[s + 1][y - 1, x]) -
               (dog_octave[s - 1][y + 1, x] - dog_octave[s - 1][y - 1, x])) / 4.0

        H = np.array([[dxx, dxy, dxs],
                      [dxy, dyy, dys],
                      [dxs, dys, dss]], dtype=np.float64)
        g = np.array([dx, dy, ds], dtype=np.float64)

        try:
            delta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta = -np.linalg.lstsq(H, g, rcond=None)[0]

        if np.all(np.abs(delta) < 0.6):
            xc = x + delta[0]
            yc = y + delta[1]
            sc = s + delta[2]
            val = dog[y, x] + 0.5 * np.dot(g, delta)
            if np.abs(val) < contrast_threshold:
                return None
            if -0.5 <= sc <= _ns - 0.5:
                sigma = _sigma0 * (2.0 ** (sc / _ns_per_octave))
                return (xc, yc, sigma)
            else:
                return None

        x += delta[0]
        y += delta[1]
        s += delta[2]

    return None


_ns = 3
_sigma0 = 1.6
_ns_per_octave = 3
_n_intervals = _ns + 3
_sigma0 = 1.6

# 计算关键点方向
def _assign_orientations(img_gray, kpts, peak_ratio=0.8, num_bins=36):
    if img_gray.dtype != np.float64:
        img = img_gray.astype(np.float64) / 255.0
    else:
        img = img_gray.copy()

    h, w = img.shape
    oriented = []
    for x, y, s in kpts:
        if x < 1 or y < 1 or x >= w - 1 or y >= h - 1:
            continue
        radius = int(np.ceil(3.0 * 1.5 * s))
        x_int = int(np.round(x))
        y_int = int(np.round(y))

        hist = np.zeros(num_bins, dtype=np.float64)
        for dy_ in range(-radius, radius + 1):
            py = y_int + dy_
            if py < 1 or py >= h - 1:
                continue
            for dx_ in range(-radius, radius + 1):
                px = x_int + dx_
                if px < 1 or px >= w - 1:
                    continue
                dx = img[py, px + 1] - img[py, px - 1]
                dy = img[py + 1, px] - img[py - 1, px]
                mag = np.sqrt(dx * dx + dy * dy)
                ori = np.arctan2(dy, dx) * 180.0 / np.pi
                if ori < 0:
                    ori += 360.0

                gauss_w = np.exp(-(dx_ * dx_ + dy_ * dy_) / (2.0 * (1.5 * s) ** 2))
                weight = mag * gauss_w
                bin_idx = int(np.floor(ori * num_bins / 360.0)) % num_bins
                hist[bin_idx] += weight

        max_val = hist.max()
        if max_val <= 0:
            continue

        thr = max_val * peak_ratio
        for bin_idx in range(num_bins):
            val = hist[bin_idx]
            if val < thr:
                continue
            prev = hist[(bin_idx - 1) % num_bins]
            next_ = hist[(bin_idx + 1) % num_bins]
            if prev < val and next_ < val:
                interp = bin_idx + 0.5 * (prev - next_) / (prev - 2.0 * val + next_)
                ori_deg = interp * 360.0 / num_bins
                if ori_deg < 0:
                    ori_deg += 360.0
                if ori_deg >= 360.0:
                    ori_deg -= 360.0
                oriented.append((x, y, s, ori_deg))

    return oriented

# 计算 SIFT 描述子
def _compute_sift_descriptor(img_gray, kp, num_angles=8, patch_size=16, grid=4):
    h, w = img_gray.shape
    x, y, s, ori = kp

    ori_rad = ori * np.pi / 180.0
    cos_t = np.cos(ori_rad)
    sin_t = np.sin(ori_rad)

    half_patch = patch_size // 2
    radius = int(np.ceil(1.414 * (half_patch + 1) * s))
    x_int = int(np.round(x))
    y_int = int(np.round(y))

    desc = np.zeros((grid, grid, num_angles), dtype=np.float64)

    for dy in range(-radius, radius + 1):
        py = y_int + dy
        if py < 1 or py >= h - 1:
            continue
        for dx in range(-radius, radius + 1):
            px = x_int + dx
            if px < 1 or px >= w - 1:
                continue

            rx = (dx * cos_t + dy * sin_t) / s
            ry = (-dx * sin_t + dy * cos_t) / s

            gx = half_patch - 1.0
            col = rx + gx
            row = ry + gx
            if col < -1.0 or row < -1.0 or col >= patch_size or row >= patch_size:
                continue

            grad_x = (img_gray[py, px + 1] - img_gray[py, px - 1]) / 2.0
            grad_y = (img_gray[py + 1, px] - img_gray[py - 1, px]) / 2.0

            mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
            angle = np.arctan2(grad_y, grad_x) * 180.0 / np.pi - ori
            if angle < 0:
                angle += 360.0
            if angle >= 360.0:
                angle -= 360.0

            gauss_w = np.exp(-(rx * rx + ry * ry) / (2.0 * (0.5 * patch_size) ** 2))
            weight = mag * gauss_w

            col = np.clip(col, -1.0, patch_size - 1)
            row = np.clip(row, -1.0, patch_size - 1)

            col0 = int(np.floor(col))
            row0 = int(np.floor(row))
            col1 = col0 + 1
            row1 = row0 + 1

            col_frac = col - col0
            row_frac = row - row0

            grid_col0 = int(np.floor(col0 * grid / patch_size))
            grid_row0 = int(np.floor(row0 * grid / patch_size))
            grid_col1 = int(np.floor(col1 * grid / patch_size))
            grid_row1 = int(np.floor(row1 * grid / patch_size))

            angle_bin = (angle / 360.0) * num_angles
            angle_idx0 = int(np.floor(angle_bin)) % num_angles
            angle_idx1 = (angle_idx0 + 1) % num_angles
            angle_frac = angle_bin - np.floor(angle_bin)

            v00 = weight * (1.0 - col_frac) * (1.0 - row_frac)
            v10 = weight * col_frac * (1.0 - row_frac)
            v01 = weight * (1.0 - col_frac) * row_frac
            v11 = weight * col_frac * row_frac

            for gc, gr, cv in [(grid_col0, grid_row0, v00),
                               (grid_col1, grid_row0, v10),
                               (grid_col0, grid_row1, v01),
                               (grid_col1, grid_row1, v11)]:
                if 0 <= gc < grid and 0 <= gr < grid and cv > 0:
                    desc[gr, gc, angle_idx0] += cv * (1.0 - angle_frac)
                    desc[gr, gc, angle_idx1] += cv * angle_frac

    desc_vec = desc.flatten()
    norm = np.linalg.norm(desc_vec, ord=2)
    if norm > 0:
        desc_vec /= norm
    desc_vec = np.clip(desc_vec, 0, 0.2)
    norm2 = np.linalg.norm(desc_vec, ord=2)
    if norm2 > 0:
        desc_vec /= norm2
    return desc_vec

# 计算金字塔层数
def _compute_octaves(img_h, img_w):
    min_dim = min(img_h, img_w)
    n_octaves = int(np.floor(np.log2(min_dim))) - 3
    n_octaves = max(1, n_octaves)
    return n_octaves

# 检测关键点
def detect_keypoints(img_gray, n_octaves=None, n_scales=_ns, sigma0=_sigma0,
                     contrast_threshold=0.02, edge_threshold=10.0):
    if img_gray.dtype != np.float64:
        img_f = img_gray.astype(np.float64) / 255.0
    else:
        img_f = img_gray.copy()

    if n_octaves is None:
        n_octaves = _compute_octaves(*img_f.shape)

    gaussian_pyr = _build_gaussian_pyramid(img_f, n_octaves, n_scales, sigma0)
    dog_pyr = _build_dog_pyramid(gaussian_pyr)

    global _ns, _ns_per_octave
    _ns = n_scales
    _ns_per_octave = n_scales

    kpts = _detect_extrema(dog_pyr, n_scales, contrast_threshold, edge_threshold)
    return kpts

# 计算描述子
def compute_descriptors(img_gray, keypoints, num_angles=8, patch_size=16, grid=4):
    oriented = _assign_orientations(img_gray, keypoints)
    descriptors = []
    valid_kpts = []
    for kp in oriented:
        desc = _compute_sift_descriptor(img_gray, kp, num_angles, patch_size, grid)
        descriptors.append(desc)
        valid_kpts.append(kp[:2])
    if len(descriptors) == 0:
        return np.empty((0, 2)), np.empty((0, grid * grid * num_angles))
    return np.array(valid_kpts, dtype=np.float64), np.array(descriptors, dtype=np.float64)

# 提取 SIFT 特征
def extract_sift(img_gray, n_octaves=None, n_scales=3, sigma0=1.6,
                 contrast_threshold=0.02, edge_threshold=10.0):
    kpts = detect_keypoints(img_gray, n_octaves, n_scales, sigma0,
                            contrast_threshold, edge_threshold)
    if len(kpts) == 0:
        return np.empty((0, 2)), np.empty((0, 128))
    pts, descs = compute_descriptors(img_gray, kpts)
    return pts, descs
