import os
import numpy as np
from .config import load_camera_config
from .calibration import undistort_image
from .features import extract_sift
from .matching import match_descriptors
from .geometry import ransac_fundamental
from .sfm import incremental_sfm
from .bundle_adjustment import bundle_adjust
from .utils import export_ply, visualize_open3d


def _read_image(filepath):
    try:
        from PIL import Image
        img = Image.open(filepath)
        return np.array(img)
    except Exception:
        pass
    try:
        import imageio
        return imageio.v3.imread(filepath)
    except Exception:
        pass
    try:
        import cv2
        return cv2.imread(filepath)
    except Exception:
        pass
    raise RuntimeError("No image reading library available (Pillow/imageio/opencv)")


def _to_gray(img):
    if img.ndim == 2:
        return img.astype(np.float64) \
            if img.dtype == np.float64 else img.astype(np.float64) / 255.0
    if img.shape[2] == 4:
        img = img[:, :, :3]
    gray = (0.299 * img[:, :, 0].astype(np.float64) +
            0.587 * img[:, :, 1].astype(np.float64) +
            0.114 * img[:, :, 2].astype(np.float64))
    if img.dtype == np.uint8:
        gray = gray / 255.0
    return gray


def _load_images(image_dir, extensions=None):
    if extensions is None:
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    files = sorted(os.listdir(image_dir))
    image_paths = []
    for f in files:
        if f.lower().endswith(extensions):
            image_paths.append(os.path.join(image_dir, f))
    return image_paths


def run_pipeline(image_dir, config_path, output_dir=None, skip_undistort=False):
    if output_dir is None:
        output_dir = os.path.join(image_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    cfg = load_camera_config(config_path)
    K = cfg['K']
    dist = cfg['distortion']
    if len(dist) < 5:
        dist = list(dist) + [0.0] * (5 - len(dist))
    dist = np.array(dist, dtype=np.float64)

    image_paths = _load_images(image_dir)
    if len(image_paths) < 2:
        raise RuntimeError(f"Need at least 2 images in {image_dir}, found {len(image_paths)}")

    print(f"Found {len(image_paths)} images")
    print(f"Step 1/5: Loading and processing images...")

    images_data = []
    for i, img_path in enumerate(image_paths):
        img = _read_image(img_path)
        if img is None:
            print(f"  Warning: Cannot read {img_path}, skipping")
            continue

        gray = _to_gray(img)

        if not skip_undistort and np.any(np.abs(dist) > 1e-10):
            gray = undistort_image(gray, K, dist)

        pts, descs = extract_sift(gray)
        images_data.append({
            'path': img_path,
            'gray': gray,
            'points': pts,
            'descriptors': descs,
        })
        print(f"  [{i}] {os.path.basename(img_path)}: {len(pts)} keypoints")

    print(f"\nStep 2/5: Feature matching...")

    for i in range(1, len(images_data)):
        matches = match_descriptors(
            images_data[i - 1]['descriptors'],
            images_data[i]['descriptors'],
        )
        images_data[i]['matches_with_prev'] = matches
        print(f"  Image {i-1} -> {i}: {len(matches)} matches")

    if len(images_data) >= 2:
        matches_01 = match_descriptors(
            images_data[0]['descriptors'],
            images_data[1]['descriptors'],
        )
        images_data[1]['matches_with_0'] = matches_01
        print(f"  Image 0 -> 1 (init): {len(matches_01)} matches")

    print(f"\nStep 3/5: Incremental SfM reconstruction...")

    sfm_state = incremental_sfm(images_data, K, dist)

    n_cameras = len(sfm_state['cameras'])
    n_points = len(sfm_state['tracks'])
    print(f"  Registered cameras: {n_cameras}/{len(images_data)}")
    print(f"  Triangulated points: {n_points}")

    cameras = sfm_state['cameras']
    tracks = sfm_state['tracks']

    points3d_list = []
    point_colors_list = []
    for track_id, track in tracks.items():
        if 'point3d' in track:
            points3d_list.append(track['point3d'])
            point_colors_list.append(track.get('color', (128, 128, 128)))

    if len(points3d_list) == 0:
        print("  No 3D points generated. Pipeline stopped.")
        return

    points3d = np.array(points3d_list, dtype=np.float64)
    point_colors = np.array(point_colors_list, dtype=np.uint8)

    print(f"\nStep 4/5: Bundle Adjustment...")

    ba_observations = {}
    for track_id, track in tracks.items():
        if 'point3d' in track and 'observations' in track:
            ba_observations[track_id] = track['observations']

    if len(ba_observations) > 0:
        refined_cameras, refined_points = bundle_adjust(
            K, cameras, points3d, ba_observations,
            max_iterations=30, huber_delta=1.0
        )
        cameras = refined_cameras
        points3d = refined_points
        print(f"  BA completed: {len(cameras)} cameras, {len(points3d)} points optimized")

    ply_path = os.path.join(output_dir, 'sparse.ply')
    export_ply(points3d, point_colors, ply_path)
    print(f"\nStep 5/5: PLY exported to {ply_path}")

    print(f"\nResult summary:")
    print(f"  Cameras: {len(cameras)}")
    print(f"  3D Points: {len(points3d)}")

    try:
        print(f"\nLaunching Open3D visualization...")
        visualize_open3d(points3d, cameras, point_colors)
    except Exception as e:
        print(f"  Visualization skipped: {e}")

    return {
        'cameras': cameras,
        'points3d': points3d,
        'ply_path': ply_path,
    }
