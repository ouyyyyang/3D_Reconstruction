import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traditional.features import extract_sift
from traditional.matching import match_descriptors
from traditional.geometry import ransac_fundamental, compute_essential
from traditional.pose import decompose_essential, select_pose
from traditional.triangulation import triangulate_dlt, filter_by_error
from traditional.sfm import incremental_sfm
from traditional.bundle_adjustment import bundle_adjust
from traditional.utils import export_ply, visualize_open3d


def generate_synthetic_data(n_points=200, n_cameras=5, img_size=(800, 600)):
    np.random.seed(42)

    K = np.array([
        [800, 0, 400],
        [0, 800, 300],
        [0, 0, 1],
    ], dtype=np.float64)

    points_3d = np.random.randn(n_points, 3) * 2.5
    points_3d[:, 2] += 12.0
    points_3d[:, 1] *= 0.3

    radius = 1.0
    cameras_gt = []
    base_angle = np.pi / 12  # 15 degrees per camera

    for i in range(n_cameras):
        angle = base_angle * i
        tx = radius * np.sin(angle)
        ty = 0.3 * np.sin(2 * angle)
        tz = radius * (np.cos(angle) - 1.0)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(0.1 * angle), -np.sin(0.1 * angle)],
            [0, np.sin(0.1 * angle), np.cos(0.1 * angle)],
        ])
        Ry = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
        R = Ry @ Rx
        t = np.array([tx, ty, tz])

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
            Image.fromarray(img).save(f'traditional/synthetic_cam_{i}.png')
        except Exception:
            pass

    return images_data, K, cameras_gt, points_3d


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
    img_pts = []
    for d in images_data:
        img_pts.append((d['points'], d['gt_pts_3d_idx']))

    for i in range(1, len(images_data)):
        prev_indices = img_pts[i - 1][1]
        curr_indices = img_pts[i][1]
        common = np.intersect1d(prev_indices, curr_indices, return_indices=True)

        matches = []
        for pi, ci in zip(common[1], common[2]):
            matches.append([pi, ci])
        matches = np.array(matches, dtype=np.int32)

        images_data[i]['matches_with_prev'] = matches
        print(f"  Image {i-1} -> {i}: {len(matches)} matches via GT")

    matches_01 = images_data[1].get('matches_with_prev', None) if 'matches_with_prev' in images_data[0] else None
    if matches_01 is not None:
        images_data[1]['matches_with_0'] = matches_01

    if len(images_data) >= 2:
        common_01 = np.intersect1d(img_pts[0][1], img_pts[1][1], return_indices=True)
        matches_01_arr = np.column_stack([common_01[1], common_01[2]])
        images_data[1]['matches_with_0'] = matches_01_arr

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

    print("\n--- Incremental SfM ---")
    sfm_state = incremental_sfm(images_data, K, dist)

    n_cameras_est = len(sfm_state['cameras'])
    n_points_est = len(sfm_state['tracks'])
    print(f"  Registered: {n_cameras_est} cameras, {n_points_est} points")
    print(f"  GT: {len(cameras_gt)} cameras, {len(points_3d_gt)} points")

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
        for track_id, track in tracks.items():
            if 'point3d' in track and 'observations' in track:
                ba_observations[track_id] = track['observations']

        refined_cameras, refined_points = bundle_adjust(
            K, cameras, points3d, ba_observations,
            max_iterations=30,
        )
        print(f"  BA: {len(refined_cameras)} cams, {len(refined_points)} pts optimized")

        print("\n--- PLY Export ---")
        colors = np.ones((len(refined_points), 3), dtype=np.uint8) * 180
        ply_path = export_ply(refined_points, colors, 'traditional/output/sparse_test.ply')
        print(f"  Exported to {ply_path}")

        print("\n--- Open3D Visualization ---")
        try:
            visualize_open3d(refined_points, refined_cameras, colors)
        except Exception as e:
            print(f"  Viz error: {e}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == '__main__':
    test_pipeline()
