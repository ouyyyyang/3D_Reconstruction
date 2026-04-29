import numpy as np

# 生成像素网格
def _meshgrid_xy(h, w):
    y = np.arange(h, dtype=np.float64)
    x = np.arange(w, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    return xx, yy

# 去畸变函数
def _distort_points(xn, yn, k1, k2, p1, p2, k3):
    r2 = xn * xn + yn * yn
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    x_radial = xn * radial
    y_radial = yn * radial
    tang_x = 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
    tang_y = p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
    return x_radial + tang_x, y_radial + tang_y

# 双线性插值
def _bilinear_interp(img, xmap, ymap):
    h, w = img.shape[:2]
    ndim = img.ndim

    x0 = np.floor(xmap).astype(np.int32)
    y0 = np.floor(ymap).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    dx = xmap - x0.astype(np.float64)
    dy = ymap - y0.astype(np.float64)

    wx00 = (1.0 - dx) * (1.0 - dy)
    wx10 = dx * (1.0 - dy)
    wx01 = (1.0 - dx) * dy
    wx11 = dx * dy

    if ndim == 2:
        result = (img[y0, x0] * wx00 +
                  img[y0, x1] * wx10 +
                  img[y1, x0] * wx01 +
                  img[y1, x1] * wx11)
    else:
        result = np.zeros_like(img, dtype=np.float64)
        for c in range(img.shape[2]):
            result[:, :, c] = (img[y0, x0, c] * wx00 +
                               img[y0, x1, c] * wx10 +
                               img[y1, x0, c] * wx01 +
                               img[y1, x1, c] * wx11)
    return result.astype(img.dtype)

# 去畸变图像
def undistort_image(img, K, dist):
    if img is None:
        return None

    if len(dist) < 5:
        dist = list(dist) + [0.0] * (5 - len(dist))

    k1, k2, p1, p2, k3 = dist
    h, w = img.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    xx, yy = _meshgrid_xy(h, w)
    xn = (xx - cx) / fx
    yn = (yy - cy) / fy

    xd, yd = _distort_points(xn, yn, k1, k2, p1, p2, k3)

    xmap = xd * fx + cx
    ymap = yd * fy + cy

    return _bilinear_interp(img, xmap, ymap)
