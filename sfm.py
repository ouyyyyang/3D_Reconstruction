import numpy as np
from .geometry import compute_fundamental_8pt, compute_essential, ransac_fundamental
from .pose import decompose_essential, select_pose
from .triangulation import triangulate_dlt, filter_by_error, reprojection_error


def _rodrigues(r):
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _projection_matrix(K, R, t):
    return K @ np.column_stack([R, t])


def _solve_pnp_dlt(pts_2d, pts_3d):
    n = len(pts_2d)
    A = np.zeros((2 * n, 12), dtype=np.float64)

    for i in range(n):
        X, Y, Z = pts_3d[i]
        u, v = pts_2d[i]
        row = 2 * i
        A[row] = [0, 0, 0, 0,
                  -X, -Y, -Z, -1,
                  v * X, v * Y, v * Z, v]
        A[row + 1] = [X, Y, Z, 1,
                      0, 0, 0, 0,
                      -u * X, -u * Y, -u * Z, -u]

    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    return P


def _decompose_projection(P, K):
    K_inv = np.linalg.inv(K)
    M = K_inv @ P

    R_bar = M[:, :3]
    t_bar = M[:, 3]

    U, _, Vt = np.linalg.svd(R_bar)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt

    scale = np.linalg.norm(R_bar) / np.linalg.norm(R)
    t = t_bar / scale

    return R, t


def _ransac_pnp(pts_2d, pts_3d, K, threshold=4.0, max_iterations=1000):
    n = len(pts_2d)
    if n < 6:
        return None, None, None

    sample_size = 6
    best_inliers = 0
    best_R, best_t = None, None
    best_mask = None
    niters = max_iterations

    for _ in range(niters):
        indices = np.random.choice(n, sample_size, replace=False)
        try:
            P = _solve_pnp_dlt(pts_2d[indices], pts_3d[indices])
            R, t = _decompose_projection(P, K)
        except (ValueError, np.linalg.LinAlgError):
            continue

        P_full = _projection_matrix(K, R, t)
        errors = np.zeros(n)
        for i in range(n):
            errors[i] = reprojection_error(P_full, pts_3d[i], pts_2d[i])

        inlier_mask = errors < threshold
        num_inliers = np.sum(inlier_mask)

        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_R, best_t = R.copy(), t.copy()
            best_mask = inlier_mask

            if num_inliers > sample_size:
                niters = min(niters, max_iterations // 2)

    if best_R is None:
        return None, None, None

    P_refined = _solve_pnp_dlt(pts_2d[best_mask], pts_3d[best_mask])
    R, t = _decompose_projection(P_refined, K)

    return R, t, best_mask


def _two_view_initialize(pts1, pts2, desc1, desc2, matches, K1, K2):
    matched_pts1 = pts1[matches[:, 0]]
    matched_pts2 = pts2[matches[:, 1]]

    F, mask = ransac_fundamental(matched_pts1, matched_pts2)
    if F is None:
        return None

    inlier_pts1 = matched_pts1[mask]
    inlier_pts2 = matched_pts2[mask]
    inlier_matches = matches[mask]

    E = compute_essential(F, K1, K2)
    candidates = decompose_essential(E)
    R, t, _ = select_pose(candidates, inlier_pts1, inlier_pts2, K1, K2)

    if R is None:
        return None

    P1 = _projection_matrix(K1, np.eye(3), np.zeros(3))
    P2 = _projection_matrix(K2, R, t)

    points_3d = triangulate_dlt(P1, P2, inlier_pts1, inlier_pts2)

    error_mask = filter_by_error(points_3d, P1, P2, inlier_pts1, inlier_pts2, max_error=4.0)
    points_3d = points_3d[error_mask]
    inlier_matches = inlier_matches[error_mask]

    cameras = [
        {'R': np.eye(3), 't': np.zeros(3), 'proj': P1},
        {'R': R, 't': t, 'proj': P2},
    ]
    registered = {0, 1}

    tracks = {}
    for idx, (i, j) in enumerate(inlier_matches):
        track_key = (0, int(i), 1, int(j))
        tracks[idx] = {
            'point3d': points_3d[idx],
            'observations': {0: int(i), 1: int(j)},
            'color': (128, 128, 128),
        }

    return {
        'cameras': cameras,
        'tracks': tracks,
        'registered': registered,
        'K': K1,
    }


def _build_match_map(matches_list):
    match_map = {}
    for img_idx, (pts, matches) in enumerate(matches_list):
        for m_idx in range(len(matches)):
            i, j = matches[m_idx]
            key = (img_idx, int(i))
            match_map[key] = int(j)
    return match_map


def incremental_sfm(images_data, K, dist_param=None,
                    min_initial_matches=50, reproj_threshold=4.0):
    n_images = len(images_data)
    if n_images < 2:
        raise ValueError("Need at least 2 images")

    K1 = K
    K2 = K

    pts1, desc1 = images_data[0]['points'], images_data[0]['descriptors']
    pts2, desc2 = images_data[1]['points'], images_data[1]['descriptors']
    matches = images_data[1].get('matches_with_0')

    if matches is None or len(matches) < min_initial_matches:
        from .matching import match_descriptors
        matches = match_descriptors(desc1, desc2)

    if len(matches) < min_initial_matches:
        raise ValueError(f"Insufficient matches for initialization: {len(matches)}")

    state = _two_view_initialize(pts1, pts2, desc1, desc2, matches, K1, K2)
    if state is None:
        raise RuntimeError("Two-view initialization failed")

    for img_idx in range(2, n_images):
        pts_new, desc_new = images_data[img_idx]['points'], images_data[img_idx]['descriptors']
        matches_with_prev = images_data[img_idx].get('matches_with_prev')
        if matches_with_prev is None:
            from .matching import match_descriptors
            matches_with_prev = match_descriptors(desc_new, images_data[img_idx - 1]['descriptors'])

        pts_2d = []
        pts_3d = []
        for m in matches_with_prev:
            i_prev, i_new = int(m[0]), int(m[1])
            for track_id, track in state['tracks'].items():
                if (img_idx - 1) in track['observations']:
                    if track['observations'][img_idx - 1] == i_prev:
                        if i_new < len(pts_new):
                            pts_2d.append(pts_new[i_new])
                            pts_3d.append(track['point3d'])
                        break

        pts_2d = np.array(pts_2d, dtype=np.float64)
        pts_3d = np.array(pts_3d, dtype=np.float64)

        if len(pts_2d) < 6:
            continue

        R, t, mask = _ransac_pnp(pts_2d, pts_3d, K, threshold=reproj_threshold)
        if R is None:
            continue

        P = _projection_matrix(K, R, t)
        state['cameras'].append({'R': R, 't': t, 'proj': P})
        state['registered'].add(img_idx)

    return state
