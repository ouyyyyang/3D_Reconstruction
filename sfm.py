import numpy as np
from .geometry import compute_fundamental_8pt, compute_essential, ransac_fundamental
from .pose import decompose_essential, select_pose
from .triangulation import triangulate_dlt, filter_by_error, reprojection_error


def _sample_color(color_img, pt):
    if color_img is None:
        return (128, 128, 128)
    x, y = int(round(pt[0])), int(round(pt[1]))
    h, w = color_img.shape[:2]
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    return tuple(int(c) for c in color_img[y, x, :3])


def _camera_center(R, t):
    return -R.T @ t


def _triangulation_angle(X, C1, C2):
    r1 = X - C1
    r2 = X - C2
    n1 = np.linalg.norm(r1)
    n2 = np.linalg.norm(r2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    cos_ang = np.dot(r1, r2) / (n1 * n2)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return np.arccos(cos_ang)


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

    centroid_2d = pts_2d.mean(axis=0)
    shifted_2d = pts_2d - centroid_2d
    avg_dist_2d = np.mean(np.sqrt(np.sum(shifted_2d ** 2, axis=1)))
    scale_2d = np.sqrt(2.0) / avg_dist_2d if avg_dist_2d > 1e-10 else 1.0
    T2 = np.array([[scale_2d, 0, -scale_2d * centroid_2d[0]],
                   [0, scale_2d, -scale_2d * centroid_2d[1]],
                   [0, 0, 1]], dtype=np.float64)

    centroid_3d = pts_3d.mean(axis=0)
    shifted_3d = pts_3d - centroid_3d
    avg_dist_3d = np.mean(np.sqrt(np.sum(shifted_3d ** 2, axis=1)))
    scale_3d = np.sqrt(3.0) / avg_dist_3d if avg_dist_3d > 1e-10 else 1.0
    T3 = np.array([[scale_3d, 0, 0, -scale_3d * centroid_3d[0]],
                   [0, scale_3d, 0, -scale_3d * centroid_3d[1]],
                   [0, 0, scale_3d, -scale_3d * centroid_3d[2]],
                   [0, 0, 0, 1]], dtype=np.float64)

    pts_2d_n = np.column_stack([shifted_2d * scale_2d, np.ones(n)])
    pts_3d_n = np.column_stack([shifted_3d * scale_3d, np.ones(n)])

    A = np.zeros((2 * n, 12), dtype=np.float64)
    for i in range(n):
        X, Y, Z, _ = pts_3d_n[i]
        u, v, _ = pts_2d_n[i]
        row = 2 * i
        A[row] = [0, 0, 0, 0,
                  -X, -Y, -Z, -1,
                  v * X, v * Y, v * Z, v]
        A[row + 1] = [X, Y, Z, 1,
                      0, 0, 0, 0,
                      -u * X, -u * Y, -u * Z, -u]

    _, _, Vt = np.linalg.svd(A)
    P_n = Vt[-1].reshape(3, 4)

    P = np.linalg.inv(T2) @ P_n @ T3
    return P


def _decompose_projection(P, K):
    K_inv = np.linalg.inv(K)
    M = K_inv @ P
    R_bar = M[:, :3]
    t = M[:, 3]

    # Handle DLT sign ambiguity: P may be -lambda * K[R|t]
    if np.linalg.det(R_bar) < 0:
        R_bar = -R_bar
        t = -t

    U, S, Vt = np.linalg.svd(R_bar)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1

    s = np.mean(S)
    if np.abs(s) > 1e-10:
        t = t / s

    return R, t


def _ransac_pnp(pts_2d, pts_3d, K, threshold=4.0, max_iterations=1000):
    n = len(pts_2d)
    if n < 4:
        return None, None, None

    try:
        import cv2
        # OpenCV solvePnPRansac with EPNP — robust to noisy real-world data
        pts_3d_cv = pts_3d.astype(np.float64).reshape(-1, 1, 3)
        pts_2d_cv = pts_2d.astype(np.float64).reshape(-1, 1, 2)
        K_cv = K.astype(np.float64)

        ret, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d_cv, pts_2d_cv, K_cv, None,
            flags=cv2.SOLVEPNP_EPNP,
            iterationsCount=max_iterations,
            reprojectionError=threshold,
            confidence=0.999,
        )
        if not ret or inliers is None or len(inliers) < 6:
            return None, None, None

        R_cv, _ = cv2.Rodrigues(rvec)
        R = R_cv.astype(np.float64)
        t = tvec.flatten().astype(np.float64)

        # Build inlier mask
        mask = np.zeros(n, dtype=bool)
        mask[inliers.flatten()] = True
        return R, t, mask

    except (ImportError, AttributeError, Exception):
        pass

    # Fallback: manual DLT-based RANSAC PnP
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
        except (ValueError, np.linalg.LinAlgError):
            continue

        errors = np.zeros(n)
        for i in range(n):
            x = P @ np.append(pts_3d[i], 1.0)
            if np.abs(x[2]) > 1e-10:
                proj = x[:2] / x[2]
                errors[i] = np.linalg.norm(proj - pts_2d[i])
            else:
                errors[i] = float('inf')

        inlier_mask = errors < threshold
        num_inliers = np.sum(inlier_mask)

        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_mask = inlier_mask.copy()

            if num_inliers > sample_size:
                inlier_ratio = num_inliers / n
                if inlier_ratio < 1.0 - 1e-10:
                    denom = np.log(1.0 - inlier_ratio ** sample_size)
                    if np.abs(denom) > 1e-12:
                        niters = int(np.ceil(np.log(1 - 0.99) / denom))
                        niters = min(niters, max_iterations)

    if best_mask is None or np.sum(best_mask) < 6:
        return None, None, None

    P_refined = _solve_pnp_dlt(pts_2d[best_mask], pts_3d[best_mask])
    R, t = _decompose_projection(P_refined, K)

    P_full = _projection_matrix(K, R, t)
    final_errors = np.zeros(n)
    for i in range(n):
        x = P_full @ np.append(pts_3d[i], 1.0)
        if np.abs(x[2]) > 1e-10:
            proj = x[:2] / x[2]
            final_errors[i] = np.linalg.norm(proj - pts_2d[i])
        else:
            final_errors[i] = float('inf')
    final_mask = final_errors < threshold

    return R, t, final_mask


def _triangulate_one_dlt(P1, P2, pt1, pt2):
    A = np.zeros((4, 4), dtype=np.float64)
    A[0] = pt1[0] * P1[2] - P1[0]
    A[1] = pt1[1] * P1[2] - P1[1]
    A[2] = pt2[0] * P2[2] - P2[0]
    A[3] = pt2[1] * P2[2] - P2[1]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]
    return X[:3].copy()


def _two_view_initialize(pts1, pts2, desc1, desc2, matches, K1, K2,
                         color_img1=None, color_img2=None):
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
        pt_obs = inlier_pts1[idx] if idx < len(inlier_pts1) else pts1[int(i)]
        color = _sample_color(color_img1, pt_obs)
        tracks[idx] = {
            'point3d': points_3d[idx],
            'observations': {0: int(i), 1: int(j)},
            'color': color,
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
                    min_initial_matches=6, reproj_threshold=10.0,
                    min_triangulation_angle=np.deg2rad(3.0)):
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
        matches = match_descriptors(desc1, desc2, ratio=0.75)

    if len(matches) < min_initial_matches:
        raise ValueError(f"Insufficient matches for initialization: {len(matches)}")

    state = _two_view_initialize(pts1, pts2, desc1, desc2, matches, K1, K2,
                                  images_data[0].get('color_img'),
                                  images_data[1].get('color_img'))
    if state is None:
        raise RuntimeError("Two-view initialization failed")

    ref_img = 0  # Dynamic reference frame, updated when overlap drops

    for img_idx in range(2, n_images):
        pts_new, desc_new = images_data[img_idx]['points'], images_data[img_idx]['descriptors']
        matches_with_prev = images_data[img_idx].get('matches_with_prev')
        if matches_with_prev is None:
            from .matching import match_descriptors
            matches_with_prev = match_descriptors(desc_new, images_data[img_idx - 1]['descriptors'], ratio=0.75)

        matches_with_ref = images_data[img_idx].get(f'matches_with_{ref_img}')
        if matches_with_ref is None:
            from .matching import match_descriptors as _md
            matches_with_ref = _md(desc_new, images_data[ref_img]['descriptors'], ratio=0.75)

        all_match_sources = [
            (img_idx - 1, matches_with_prev),
            (ref_img, matches_with_ref),
        ]
        if (img_idx - 2) in state['registered']:
            matches_with_prev2 = images_data[img_idx].get('matches_with_prev2')
            if matches_with_prev2 is not None:
                all_match_sources.append((img_idx - 2, matches_with_prev2))
        if (img_idx - 3) in state['registered']:
            matches_with_prev3 = images_data[img_idx].get('matches_with_prev3')
            if matches_with_prev3 is not None:
                all_match_sources.append((img_idx - 3, matches_with_prev3))

        pts_2d_raw = []
        pts_3d_raw = []
        seen_tracks = set()
        track_list = []
        for src_img_idx, matches_src in all_match_sources:
            for m in matches_src:
                i_src, i_new = int(m[0]), int(m[1])
                for track_id, track in state['tracks'].items():
                    if src_img_idx in track['observations']:
                        if track['observations'][src_img_idx] == i_src:
                            if track_id not in seen_tracks and i_new < len(pts_new):
                                seen_tracks.add(track_id)
                                track_list.append((track_id, i_new))
                                pts_2d_raw.append(pts_new[i_new])
                                pts_3d_raw.append(track['point3d'])
                            break

        pts_2d = np.array(pts_2d_raw, dtype=np.float64)
        pts_3d = np.array(pts_3d_raw, dtype=np.float64)
        match_info = {idx: (tid, i_new) for idx, (tid, i_new) in enumerate(track_list)}

        print(f"  [Cam {img_idx}] 2D-3D correspondences: {len(pts_2d)} (unique tracks)")

        if len(pts_2d) < 6:
            print(f"  [Cam {img_idx}] SKIPPED: insufficient correspondences (< 6)")
            continue

        R, t, mask = _ransac_pnp(pts_2d, pts_3d, K, threshold=reproj_threshold)
        if R is None:
            print(f"  [Cam {img_idx}] PnP FAILED: no valid pose found")
            continue

        n_inliers = np.sum(mask) if mask is not None else 0
        print(f"  [Cam {img_idx}] PnP OK: {n_inliers} inliers / {len(pts_2d)} pts")

        P = _projection_matrix(K, R, t)
        state['cameras'].append({'R': R, 't': t, 'proj': P})
        state['registered'].add(img_idx)

        tracked_inds = {i for i in match_info.keys()}
        for i in tracked_inds:
            track_id, i_new = match_info[i]
            state['tracks'][track_id]['observations'][img_idx] = i_new

        n_tracks_before_new = max(state['tracks'].keys()) if state['tracks'] else -1

        prev_cam_idx = len(state['cameras']) - 2
        P_prev = state['cameras'][prev_cam_idx]['proj']
        P_new = P
        pts_prev = images_data[img_idx - 1]['points']
        max_track_id = max(state['tracks'].keys()) if state['tracks'] else -1

        img_to_cam = {reg_img: cam_idx
                      for cam_idx, reg_img in enumerate(sorted(state['registered']))}
        ref_cam_idx = img_to_cam.get(ref_img, 0)
        P_ref = state['cameras'][ref_cam_idx]['proj']
        pts_ref = images_data[ref_img]['points']

        prev_observed = {}
        for track_id, track in state['tracks'].items():
            if (img_idx - 1) in track['observations']:
                prev_observed[track['observations'][img_idx - 1]] = True

        for m_idx, m in enumerate(matches_with_prev):
            i_prev, i_new = int(m[0]), int(m[1])
            if i_prev in prev_observed:
                continue
            if i_new >= len(pts_new) or i_prev >= len(pts_prev):
                continue
            pt1 = pts_prev[i_prev]
            pt2 = pts_new[i_new]
            try:
                X = _triangulate_one_dlt(P_prev, P_new, pt1, pt2)
            except (ValueError, np.linalg.LinAlgError):
                continue
            C_prev = _camera_center(state['cameras'][prev_cam_idx]['R'],
                                    state['cameras'][prev_cam_idx]['t'])
            C_new = _camera_center(R, t)
            if _triangulation_angle(X, C_prev, C_new) < min_triangulation_angle:
                continue
            x1_cam = P_prev @ np.append(X, 1.0)
            x2_cam = P_new @ np.append(X, 1.0)
            if x1_cam[2] <= 0 or x2_cam[2] <= 0:
                continue
            err1 = reprojection_error(P_prev, X, pt1)
            err2 = reprojection_error(P_new, X, pt2)
            if err1 > reproj_threshold or err2 > reproj_threshold:
                continue
            max_track_id += 1
            state['tracks'][max_track_id] = {
                'point3d': X,
                'observations': {img_idx - 1: i_prev, img_idx: i_new},
                'color': _sample_color(images_data[img_idx].get('color_img'),
                                        pts_new[i_new]),
            }
            prev_observed[i_prev] = True

        obs_ref_set = {}
        for track_id, track in state['tracks'].items():
            if ref_img in track['observations']:
                obs_ref_set[track['observations'][ref_img]] = True

        for m in matches_with_ref:
            i_ref, i_new = int(m[0]), int(m[1])
            if i_ref in obs_ref_set:
                continue
            if i_new >= len(pts_new) or i_ref >= len(pts_ref):
                continue
            pt1 = pts_ref[i_ref]
            pt2 = pts_new[i_new]
            try:
                X = _triangulate_one_dlt(P_ref, P_new, pt1, pt2)
            except (ValueError, np.linalg.LinAlgError):
                continue
            C_ref = _camera_center(state['cameras'][ref_cam_idx]['R'],
                                   state['cameras'][ref_cam_idx]['t'])
            C_new = _camera_center(R, t)
            if _triangulation_angle(X, C_ref, C_new) < min_triangulation_angle:
                continue
            x1_cam = P_ref @ np.append(X, 1.0)
            x2_cam = P_new @ np.append(X, 1.0)
            if x1_cam[2] <= 0 or x2_cam[2] <= 0:
                continue
            err1 = reprojection_error(P_ref, X, pt1)
            err2 = reprojection_error(P_new, X, pt2)
            if err1 > reproj_threshold or err2 > reproj_threshold:
                continue
            max_track_id += 1
            state['tracks'][max_track_id] = {
                'point3d': X,
                'observations': {ref_img: i_ref, img_idx: i_new},
                'color': _sample_color(images_data[img_idx].get('color_img'),
                                        pts_new[i_new]),
            }
            obs_ref_set[i_ref] = True

        if (img_idx - 2) in img_to_cam and 'matches_with_prev2' in images_data[img_idx]:
            matches_with_prev2 = images_data[img_idx]['matches_with_prev2']
            prev2_cam_idx = img_to_cam[img_idx - 2]
            P_prev2 = state['cameras'][prev2_cam_idx]['proj']
            pts_prev2 = images_data[img_idx - 2]['points']
            prev2_observed = {}
            for track_id, track in state['tracks'].items():
                if (img_idx - 2) in track['observations']:
                    prev2_observed[track['observations'][img_idx - 2]] = True
            for m in matches_with_prev2:
                i_p2, i_new = int(m[0]), int(m[1])
                if i_p2 in prev2_observed:
                    continue
                if i_new >= len(pts_new) or i_p2 >= len(pts_prev2):
                    continue
                pt1 = pts_prev2[i_p2]
                pt2 = pts_new[i_new]
                try:
                    X = _triangulate_one_dlt(P_prev2, P_new, pt1, pt2)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                C_prev2 = _camera_center(state['cameras'][prev2_cam_idx]['R'],
                                         state['cameras'][prev2_cam_idx]['t'])
                C_new = _camera_center(R, t)
                if _triangulation_angle(X, C_prev2, C_new) < min_triangulation_angle:
                    continue
                x1_cam = P_prev2 @ np.append(X, 1.0)
                x2_cam = P_new @ np.append(X, 1.0)
                if x1_cam[2] <= 0 or x2_cam[2] <= 0:
                    continue
                err1 = reprojection_error(P_prev2, X, pt1)
                err2 = reprojection_error(P_new, X, pt2)
                if err1 > reproj_threshold or err2 > reproj_threshold:
                    continue
                max_track_id += 1
                state['tracks'][max_track_id] = {
                    'point3d': X,
                    'observations': {img_idx - 2: i_p2, img_idx: i_new},
                    'color': _sample_color(images_data[img_idx].get('color_img'),
                                        pts_new[i_new]),
                }
                prev2_observed[i_p2] = True

        n_new_triangulated = max(state['tracks'].keys()) - max(n_tracks_before_new, -1)
        print(f"  [Cam {img_idx}] New points triangulated: {max(0, n_new_triangulated)}")

        # Periodic reference frame update
        n_ref_matches = len(matches_with_ref) if matches_with_ref is not None else 0
        if n_ref_matches < 50 and (img_idx - ref_img) > 5:
            for candidate in range(img_idx - 2, ref_img, -1):
                if candidate in state['registered']:
                    print(f"  [Ref] Switching reference frame: {ref_img} -> {candidate}")
                    ref_img = candidate
                    break

    return state
