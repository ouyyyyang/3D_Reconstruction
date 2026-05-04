import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from traditional.features import extract_sift
from traditional.matching import match_descriptors
from traditional.geometry import ransac_fundamental, compute_essential
from traditional.pose import decompose_essential, select_pose
from traditional.triangulation import triangulate_dlt, filter_by_error
from traditional.sfm import incremental_sfm
from traditional.bundle_adjustment import bundle_adjust
from traditional.utils import export_ply, visualize_open3d


def _gt_matches_for_pair(img_a, img_b):
    """Return (N,2) match array using GT point index correspondences."""
    idx_a = img_a['gt_pts_3d_idx']
    idx_b = img_b['gt_pts_3d_idx']
    common = np.intersect1d(idx_a, idx_b, return_indices=True)
    return np.column_stack([common[1], common[2]])


def generate_synthetic_data(n_points=200, n_cameras=5, img_size=(800, 600)):
    np.random.seed(42)

    K = np.array([
        [800, 0, 400],
        [0, 800, 300],
        [0, 0, 1],
    ], dtype=np.float64)

    # Points spread around origin, roughly spherical
    points_3d = np.random.randn(n_points, 3) * 1.5

    # Cameras orbit around the point cloud, always looking at origin
    orbit_radius = 5.0
    cameras_gt = []
    world_up = np.array([0.0, 1.0, 0.0])

    for i in range(n_cameras):
        angle = 2.0 * np.pi * i / n_cameras
        eye = np.array([orbit_radius * np.sin(angle), 0.0, orbit_radius * np.cos(angle)])
        target = np.array([0.0, 0.0, 0.0])

        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        # Guard against gimbal lock: if forward || world_up, use different up
        if np.abs(np.dot(forward, world_up)) > 0.999:
            world_up = np.array([0.0, 0.0, 1.0])

        right = np.cross(world_up, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)

        # R maps world→camera: rows = camera axes in world coords
        R = np.vstack([right, up, forward])
        t = -R @ eye

        cameras_gt.append({'R': R, 't': t})

    images_data = []
    for i, cam in enumerate(cameras_gt):
        R, t = cam['R'], cam['t']
        P = K @ np.column_stack([R, t])

        img = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)

        pts_2d_list = []
        for X in points_3d:
            x = P @ np.append(X, 1.0)
            u = x[0] / x[2]
            v = x[1] / x[2]
            pts_2d_list.append([u, v])

        pts_2d = np.array(pts_2d_list, dtype=np.float64)

        mask = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_size[0]) & \
               (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_size[1]) & \
               ((R @ points_3d.T).T[:, 2] + t[2] > 0)

        for j, (u, v) in enumerate(pts_2d[mask]):
            u_int, v_int = np.clip(int(u), 0, img_size[0] - 1), np.clip(int(v), 0, img_size[1] - 1)
            x0, y0 = max(0, u_int - 2), max(0, v_int - 2)
            x1, y1 = min(img_size[0], u_int + 3), min(img_size[1], v_int + 3)
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    if (xx - u_int)**2 + (yy - v_int)**2 <= 4:
                        img[yy, xx] = 255

        images_data.append({
            'path': f'synthetic_{i}.png',
            'gray': img.astype(np.float64) / 255.0,
            'points': pts_2d[mask],
            'descriptors': np.eye(np.sum(mask), 128),
            'gt_pts_3d_idx': np.where(mask)[0],
        })

        try:
            from PIL import Image
            out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
            os.makedirs(out_dir, exist_ok=True)
            #Image.fromarray(img).save(os.path.join(out_dir, f'synthetic_cam_{i}.png'))
        except Exception:
            pass

    return images_data, K, cameras_gt, points_3d


def evaluate_reconstruction(sfm_state, cameras_gt, points_3d_gt, images_data, K,
                            ba_cameras=None, ba_points=None):
    """定量评估重建结果 vs GT"""
    registered = sfm_state['registered']
    n_gt = len(cameras_gt)
    registered_sorted = sorted(registered)
    img_to_cam = {img_idx: cam_idx for cam_idx, img_idx in enumerate(registered_sorted)}
    est_cameras = ba_cameras if ba_cameras is not None else sfm_state['cameras']
    n_reg = len(registered_sorted)

    print("\n" + "=" * 60)
    print("QUANTITATIVE EVALUATION")
    print("=" * 60)

    # --- 1. 注册率 ---
    print(f"\n[1] Registration Rate: {n_reg}/{n_gt} ({n_reg/n_gt*100:.1f}%)")
    for i in range(n_gt):
        status = "OK" if i in registered else "MISSING"
        print(f"    Cam {i}: {status}")

    # --- 2. 相似变换对齐（用相机中心做 Procrustes） ---
    est_centers = []
    gt_centers = []
    for img_idx in registered_sorted:
        if img_idx >= n_gt:
            continue
        cam_idx = img_to_cam[img_idx]
        R_est = est_cameras[cam_idx]['R']
        t_est = est_cameras[cam_idx]['t']
        C_est = -R_est.T @ t_est  # camera center in world coords

        R_gt = cameras_gt[img_idx]['R']
        t_gt = cameras_gt[img_idx]['t']
        C_gt = -R_gt.T @ t_gt

        est_centers.append(C_est)
        gt_centers.append(C_gt)

    est_centers = np.array(est_centers)
    gt_centers = np.array(gt_centers)

    # Procrustes: find R_align, s_align, t_align
    centroid_est = est_centers.mean(axis=0)
    centroid_gt = gt_centers.mean(axis=0)
    A = (est_centers - centroid_est).T @ (gt_centers - centroid_gt)
    U, _, Vt = np.linalg.svd(A)
    R_align = Vt.T @ U.T
    if np.linalg.det(R_align) < 0:
        R_align = Vt.T @ np.diag([1, 1, -1]) @ U.T
    s_num = np.sum((gt_centers - centroid_gt) * ((R_align @ (est_centers - centroid_est).T).T))
    s_den = np.sum((est_centers - centroid_est) ** 2)
    s_align = s_num / s_den if s_den > 1e-10 else 1.0
    t_align = centroid_gt - s_align * R_align @ centroid_est

    print(f"\n[2] Similarity alignment: s={s_align:.4f}, R_align det={np.linalg.det(R_align):.1f}")

    # --- 3. 相机位姿误差 ---
    print(f"\n[3] Camera Pose Errors (after similarity alignment):")
    rot_errs = []
    t_errs = []
    for i in range(n_gt):
        if i not in img_to_cam:
            continue
        cam_idx = img_to_cam[i]
        R_est_raw = est_cameras[cam_idx]['R']
        t_est_raw = est_cameras[cam_idx]['t']

        # Apply alignment to camera center
        C_est = -R_est_raw.T @ t_est_raw
        C_aligned = s_align * R_align @ C_est + t_align

        # Apply alignment to rotation: R_aligned = R_est_raw @ R_align.T
        # (R_align maps world_est → world_gt; R_est maps world→cam in est frame)
        R_est = R_est_raw @ R_align.T
        t_est = -R_est @ C_aligned

        R_gt = cameras_gt[i]['R']
        t_gt = cameras_gt[i]['t']
        C_gt = -R_gt.T @ t_gt

        # 旋转误差 (度)
        R_diff = R_gt @ R_est.T
        trace_val = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
        rot_err = np.arccos(trace_val) * 180.0 / np.pi

        # 中心位置相对误差
        dist_gt = np.linalg.norm(C_gt)
        if dist_gt > 1e-10:
            c_err = np.linalg.norm(C_aligned - C_gt) / dist_gt
        else:
            c_err = np.linalg.norm(C_aligned - C_gt)

        rot_errs.append(rot_err)
        t_errs.append(c_err)
        flag = "  <<< BAD" if rot_err > 5.0 or c_err > 0.1 else ""
        print(f"    Cam {i}: rot={rot_err:.2f}°, center_err={c_err:.4f}{flag}")

    if rot_errs:
        print(f"    Mean rot error: {np.mean(rot_errs):.2f}°")
    if t_errs:
        print(f"    Mean center error: {np.mean(t_errs):.4f}")

    # --- 4. 重投影误差 ---
    print(f"\n[4] Reprojection Errors (GT points → aligned estimated cams):")
    all_reproj = []
    for img_idx in registered_sorted:
        if img_idx >= n_gt:
            continue
        cam_idx = img_to_cam[img_idx]
        R_est_raw = est_cameras[cam_idx]['R']
        t_est_raw = est_cameras[cam_idx]['t']
        C_est = -R_est_raw.T @ t_est_raw
        C_aligned = s_align * R_align @ C_est + t_align
        R_est = R_est_raw @ R_align.T
        t_est = -R_est @ C_aligned
        P = K @ np.column_stack([R_est, t_est])

        if 'gt_pts_3d_idx' not in images_data[img_idx]:
            continue
        gt_indices = images_data[img_idx]['gt_pts_3d_idx']
        pts_2d_obs = images_data[img_idx]['points']

        errs = []
        for feat_i, gt_i in enumerate(gt_indices):
            if feat_i >= len(pts_2d_obs):
                break
            X_gt = points_3d_gt[gt_i]
            x_proj = P @ np.append(X_gt, 1.0)
            if x_proj[2] > 1e-10:
                u_proj = x_proj[:2] / x_proj[2]
                err = np.linalg.norm(u_proj - pts_2d_obs[feat_i])
                errs.append(err)
        if errs:
            mean_e = np.mean(errs)
            all_reproj.extend(errs)
            print(f"    Cam {img_idx}: {len(errs)} pts, mean_reproj={mean_e:.2f} px")
        else:
            print(f"    Cam {img_idx}: no valid projections")

    if all_reproj:
        print(f"    Overall mean reprojection: {np.mean(all_reproj):.2f} px")

    # --- 5. 3D 点误差（通过 track 中的 GT 对应） ---
    print(f"\n[5] 3D Point Errors (Procrustes aligned):")
    tracks = sfm_state['tracks']
    est_pts = []
    gt_pts = []
    for track_id, track in tracks.items():
        if 'point3d' not in track or 'observations' not in track:
            continue
        for img_idx, feat_idx in track['observations'].items():
            if img_idx in images_data and 'gt_pts_3d_idx' in images_data[img_idx]:
                gt_map = images_data[img_idx]['gt_pts_3d_idx']
                if feat_idx < len(gt_map):
                    est_pts.append(track['point3d'])
                    gt_pts.append(points_3d_gt[gt_map[feat_idx]])
                    break  # 每个 track 只取第一次对应

    if len(est_pts) >= 3:
        est_arr_raw = np.array(est_pts, dtype=np.float64)
        # Align estimated points to GT frame
        est_arr = (s_align * R_align @ est_arr_raw.T).T + t_align
        gt_arr = np.array(gt_pts, dtype=np.float64)

        errors = np.linalg.norm(est_arr - gt_arr, axis=1)
        print(f"    {len(est_arr)} point pairs (after similarity alignment)")
        print(f"    RMSE: {np.sqrt(np.mean(errors**2)):.4f}")
        print(f"    Median error: {np.median(errors):.4f}")
        print(f"    Max error: {np.max(errors):.4f}")

    print("=" * 60)


def test_pipeline():
    print("=" * 60)
    print("Synthetic Data Test - Traditional SfM Pipeline")
    print("=" * 60)

    images_data, K, cameras_gt, points_3d_gt = generate_synthetic_data(
        n_points=200, n_cameras=5
    )

    dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    print(f"\nImages: {len(images_data)}")
    for i, d in enumerate(images_data):
        print(f"  Image {i}: {len(d['points'])} projected points")

    print("\n--- Feature Matching (using GT correspondences) ---")

    # Provide GT matches for all image pairs that incremental_sfm needs
    for i in range(1, len(images_data)):
        # Adjacent pair: img i-1 → img i
        m = _gt_matches_for_pair(images_data[i - 1], images_data[i])
        images_data[i]['matches_with_prev'] = m
        print(f"  Image {i-1} -> {i} (prev): {len(m)} matches")

        # Pair with image 0
        m0 = _gt_matches_for_pair(images_data[0], images_data[i])
        images_data[i]['matches_with_0'] = m0
        print(f"  Image 0 -> {i}: {len(m0)} matches")

        # Prev2: img i-2 → img i
        if i >= 2:
            m2 = _gt_matches_for_pair(images_data[i - 2], images_data[i])
            images_data[i]['matches_with_prev2'] = m2
            print(f"  Image {i-2} -> {i} (prev2): {len(m2)} matches")

        # Prev3: img i-3 → img i
        if i >= 3:
            m3 = _gt_matches_for_pair(images_data[i - 3], images_data[i])
            images_data[i]['matches_with_prev3'] = m3
            print(f"  Image {i-3} -> {i} (prev3): {len(m3)} matches")

    print("\n--- Geometry Verification (2-View) ---")
    pts0, pts1 = images_data[0]['points'], images_data[1]['points']
    matches_m = images_data[1]['matches_with_0']

    matched_pts0 = pts0[matches_m[:, 0]]
    matched_pts1 = pts1[matches_m[:, 1]]

    F, fmask = ransac_fundamental(matched_pts0, matched_pts1, threshold=2.0)
    print(f"  Fundamental matrix estimated: {F is not None}")
    print(f"  Inliers: {np.sum(fmask) if fmask is not None else 0}/{len(matched_pts0)}")

    E = compute_essential(F, K, K)
    print(f"  Essential matrix rank: {np.linalg.matrix_rank(E)}")

    candidates = decompose_essential(E)
    R_est, t_est, pts3d_init = select_pose(candidates, matched_pts0[fmask], matched_pts1[fmask], K, K)
    print(f"  Selected pose - R:\n{R_est}")
    print(f"  Selected pose - t: {t_est}")

    R_gt = cameras_gt[1]['R']
    t_gt = cameras_gt[1]['t']
    print(f"  GT pose - R:\n{R_gt}")
    print(f"  GT pose - t: {t_gt}")

    # Quick check: triangulate the initial pair and verify reprojection
    P1_init = K @ np.column_stack([np.eye(3), np.zeros(3)])
    P2_init = K @ np.column_stack([R_est, t_est])
    inlier_pts1 = matched_pts0[fmask]
    inlier_pts2 = matched_pts1[fmask]
    X_init = triangulate_dlt(P1_init, P2_init, inlier_pts1, inlier_pts2)
    max_init_error = 0
    for idx in range(len(X_init)):
        x1 = P1_init @ np.append(X_init[idx], 1.0)
        x2 = P2_init @ np.append(X_init[idx], 1.0)
        if x1[2] > 1e-10 and x2[2] > 1e-10:
            e1 = np.linalg.norm(x1[:2]/x1[2] - inlier_pts1[idx])
            e2 = np.linalg.norm(x2[:2]/x2[2] - inlier_pts2[idx])
            max_init_error = max(max_init_error, e1, e2)
    print(f"  Initial triangulation max reproj error: {max_init_error:.6f} px")
    print(f"  {'PASS' if max_init_error < 1.0 else 'FAIL — initial 3D points are WRONG'}")

    print("\n--- Incremental SfM ---")
    sfm_state = incremental_sfm(images_data, K, dist)

    n_cameras_est = len(sfm_state['cameras'])
    n_points_est = len(sfm_state['tracks'])
    print(f"  Registered: {n_cameras_est} cameras, {n_points_est} points")
    print(f"  GT: {len(cameras_gt)} cameras, {len(points_3d_gt)} points")

    # --- 评估：SfM 之后（BA 之前） ---
    evaluate_reconstruction(sfm_state, cameras_gt, points_3d_gt, images_data, K)

    cameras = sfm_state['cameras']
    tracks = sfm_state['tracks']

    points3d_list = []
    for track_id, track in tracks.items():
        if 'point3d' in track:
            points3d_list.append(track['point3d'])

    if len(points3d_list) > 0:
        points3d = np.array(points3d_list, dtype=np.float64)

        print("\n--- Bundle Adjustment ---")
        ba_observations = {}
        registered_sorted = sorted(sfm_state['registered'])
        img_to_cam = {img_idx: cam_idx for cam_idx, img_idx in enumerate(registered_sorted)}
        for track_id, track in tracks.items():
            if 'point3d' not in track or 'observations' not in track:
                continue
            obs_with_coords = {}
            for img_idx, feat_idx in track['observations'].items():
                if img_idx in img_to_cam:
                    cam_idx = img_to_cam[img_idx]
                    obs_with_coords[cam_idx] = images_data[img_idx]['points'][feat_idx].copy()
            if len(obs_with_coords) >= 2:
                ba_observations[track_id] = obs_with_coords

        refined_cameras, refined_points = bundle_adjust(
            K, cameras, points3d, ba_observations,
            max_iterations=30,
        )
        print(f"  BA: {len(refined_cameras)} cams, {len(refined_points)} pts optimized")

        # --- 评估：BA 之后 ---
        evaluate_reconstruction(sfm_state, cameras_gt, points_3d_gt, images_data, K,
                                ba_cameras=refined_cameras, ba_points=refined_points)

        print("\n--- PLY Export ---")
        colors = np.ones((len(refined_points), 3), dtype=np.uint8) * 180
        out_ply = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'sparse_test.ply')
        ply_path = export_ply(refined_points, colors, out_ply)
        print(f"  Exported to {ply_path}")

        print("\n--- Open3D Visualization ---")
        try:
            visualize_open3d(refined_points, refined_cameras, colors)
        except Exception as e:
            print(f"  Viz error: {e}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


def test_init_pair_gap(n_cameras=12, init_gap=1):
    """Test: N cameras on full circle. init_gap=1 means adjacent init pair.
    Larger init_gap selects a wider-baseline initial pair.
    Returns number of registered cameras."""
    images_data, K, cameras_gt, points_3d_gt = generate_synthetic_data(
        n_points=500, n_cameras=n_cameras
    )
    dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # Set up all GT matches for the pipeline
    for i in range(1, n_cameras):
        images_data[i]['matches_with_prev'] = _gt_matches_for_pair(images_data[i - 1], images_data[i])
        images_data[i]['matches_with_0'] = _gt_matches_for_pair(images_data[0], images_data[i])
        if i >= 2:
            images_data[i]['matches_with_prev2'] = _gt_matches_for_pair(images_data[i - 2], images_data[i])
        if i >= 3:
            images_data[i]['matches_with_prev3'] = _gt_matches_for_pair(images_data[i - 3], images_data[i])

    # Add noise to simulate real feature detection error (~1.0 px std)
    rng = np.random.RandomState(42)
    for d in images_data:
        noise = rng.randn(len(d['points']), 2) * 1.0
        d['points'] = d['points'] + noise

    # Inject outlier matches (~10% of matches are wrong)
    for i in range(1, n_cameras):
        if 'matches_with_prev' in images_data[i]:
            m = images_data[i]['matches_with_prev']
            n_out = max(1, len(m) // 10)
            bad_idx = rng.choice(len(m), n_out, replace=False)
            m[bad_idx, 0] = rng.randint(0, len(images_data[i-1]['points']), n_out)

    # Apply init_gap: swap images_data[1] with images_data[init_gap]
    if init_gap != 1 and init_gap < n_cameras:
        images_data[1], images_data[init_gap] = images_data[init_gap], images_data[1]
        # Rebuild matches for the new order
        images_data[1]['matches_with_0'] = _gt_matches_for_pair(images_data[0], images_data[1])
        for i in range(1, n_cameras):
            images_data[i]['matches_with_prev'] = _gt_matches_for_pair(images_data[i - 1], images_data[i])
            images_data[i]['matches_with_0'] = _gt_matches_for_pair(images_data[0], images_data[i])
            if i >= 2:
                images_data[i]['matches_with_prev2'] = _gt_matches_for_pair(images_data[i - 2], images_data[i])
            if i >= 3:
                images_data[i]['matches_with_prev3'] = _gt_matches_for_pair(images_data[i - 3], images_data[i])

    sfm_state = incremental_sfm(images_data, K, dist)
    n_reg = len(sfm_state['registered'])
    print(f"  init_gap={init_gap}: {n_reg}/{n_cameras} registered")
    return n_reg


def test_init_pair_comparison():
    """Tracer bullet: verify that wider init baseline improves registration."""
    print("=" * 60)
    print("Init Pair Baseline Test")
    print("=" * 60)

    n_cameras = 12  # 30° spacing, full 360° circle

    # Adjacent init (5° equivalent in real data)
    n_adj = test_init_pair_gap(n_cameras, init_gap=1)
    # Wide init (90° apart)
    n_wide = test_init_pair_gap(n_cameras, init_gap=3)

    print(f"\nResult: adjacent init={n_adj}/{n_cameras}, wide init={n_wide}/{n_cameras}")
    if n_wide > n_adj:
        print("PASS: wider baseline improves registration")
    elif n_wide == n_cameras:
        print("PASS: wide baseline achieves full registration")
    else:
        print("UNEXPECTED: check init pair logic")

    print("=" * 60)


if __name__ == '__main__':
    test_pipeline()
    test_init_pair_comparison()
