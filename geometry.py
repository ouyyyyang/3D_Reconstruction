import numpy as np


def _normalize_points(pts):
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    avg_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))
    scale = np.sqrt(2.0) / avg_dist if avg_dist > 1e-10 else 1.0
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    pts_norm = np.column_stack([shifted * scale, np.ones(len(pts))])
    return pts_norm[:, :2], T


def compute_fundamental_8pt(pts1, pts2):
    n = len(pts1)
    if n < 8:
        raise ValueError("Need at least 8 point correspondences")

    pts1_n, T1 = _normalize_points(pts1)
    pts2_n, T2 = _normalize_points(pts2)

    A = np.empty((n, 9), dtype=np.float64)
    for i in range(n):
        x1, y1 = pts1_n[i]
        x2, y2 = pts2_n[i]
        A[i] = [x2 * x1, x2 * y1, x2,
                y2 * x1, y2 * y1, y2,
                x1, y1, 1.0]

    _, _, Vt = np.linalg.svd(A)
    F_vec = Vt[-1]
    F_mat = F_vec.reshape(3, 3)

    U, S, Vt = np.linalg.svd(F_mat)
    S[-1] = 0
    F_mat = U @ np.diag(S) @ Vt

    F = T2.T @ F_mat @ T1
    F /= F[2, 2] if np.abs(F[2, 2]) > 1e-10 else 1.0

    return F


def compute_essential(F, K1, K2):
    E = K2.T @ F @ K1
    U, _, Vt = np.linalg.svd(E)
    S = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    E = U @ np.diag(S) @ Vt
    return E


def _sampson_distance(F, pts1, pts2):
    n = len(pts1)
    p1 = np.column_stack([pts1, np.ones(n)])
    p2 = np.column_stack([pts2, np.ones(n)])

    Fp1 = F @ p1.T
    FTp2 = F.T @ p2.T

    epiline2 = (Fp1[0] * p2[:, 0] + Fp1[1] * p2[:, 1] + Fp1[2]) ** 2
    epiline1 = (FTp2[0] * p1[:, 0] + FTp2[1] * p1[:, 1] + FTp2[2]) ** 2

    denom = Fp1[0] ** 2 + Fp1[1] ** 2 + FTp2[0] ** 2 + FTp2[1] ** 2
    denom = np.maximum(denom, 1e-12)

    return epiline2 / denom


def ransac_fundamental(pts1, pts2, threshold=1.0, max_iterations=2000, confidence=0.99):
    n = len(pts1)
    if n < 8:
        return None, None

    best_inliers = 0
    best_F = None
    best_mask = None

    sample_size = 8
    inlier_ratio = 0.0
    niters = max_iterations

    for iteration in range(niters):
        indices = np.random.choice(n, sample_size, replace=False)
        try:
            F = compute_fundamental_8pt(pts1[indices], pts2[indices])
        except (ValueError, np.linalg.LinAlgError):
            continue

        errors = _sampson_distance(F, pts1, pts2)
        inlier_mask = errors < (threshold * threshold)
        num_inliers = np.sum(inlier_mask)

        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_F = F
            best_mask = inlier_mask

            inlier_ratio = max(num_inliers / n, 1e-10)
            if inlier_ratio < 1.0 - 1e-10:
                denom = np.log(1.0 - inlier_ratio ** sample_size)
                if np.abs(denom) > 1e-12:
                    niters = int(np.ceil(
                        np.log(1 - confidence) / denom))
                    niters = min(niters, max_iterations)
            else:
                niters = 0

    if best_F is None:
        return None, None

    F_final = compute_fundamental_8pt(pts1[best_mask], pts2[best_mask])
    return F_final, best_mask
