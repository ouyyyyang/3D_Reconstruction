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


def _resize_image(img, max_dim):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img, 1.0
    try:
        from PIL import Image
        scale = max_dim / max(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        pil_img = Image.fromarray(img)
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        return np.array(resized), scale
    except Exception:
        pass
    try:
        import cv2
        scale = max_dim / max(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    except Exception:
        pass
    return img, 1.0


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

    import re
    def _sort_key(f):
        name = os.path.splitext(f)[0]
        nums = re.findall(r'\d+', name)
        return (int(nums[0]) if nums else 0, f)
    files = sorted(os.listdir(image_dir), key=_sort_key)
    image_paths = []
    for f in files:
        if f.lower().endswith(extensions):
            image_paths.append(os.path.join(image_dir, f))
    return image_paths


def run_pipeline(image_dir, config_path, output_dir=None, skip_undistort=False, max_image_dim=1276):
    if output_dir is None:
        output_dir = os.path.join(image_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    cfg = load_camera_config(config_path)
    K = cfg['K'].copy()
    dist = cfg['distortion']
    if len(dist) < 5:
        dist = list(dist) + [0.0] * (5 - len(dist))
    dist = np.array(dist, dtype=np.float64)

    image_paths = _load_images(image_dir)
    if len(image_paths) < 2:
        raise RuntimeError(f"Need at least 2 images in {image_dir}, found {len(image_paths)}")

    print(f"Found {len(image_paths)} images")
    print(f"Step 1/5: Loading and processing images...")

    resized_K = None
    images_data = []
    for i, img_path in enumerate(image_paths):
        img = _read_image(img_path)
        if img is None:
            print(f"  Warning: Cannot read {img_path}, skipping")
            continue

        img_resized, scale = _resize_image(img, max_image_dim)
        if abs(scale - 1.0) > 1e-6 and resized_K is None:
            resized_K = K.copy()
            resized_K[0, 0] *= scale
            resized_K[1, 1] *= scale
            resized_K[0, 2] *= scale
            resized_K[1, 2] *= scale
            print(f"  Resized images: {img.shape[1]}x{img.shape[0]} -> {img_resized.shape[1]}x{img_resized.shape[0]} (scale={scale:.3f})")

        gray = _to_gray(img_resized)

        if not skip_undistort and np.any(np.abs(dist) > 1e-10):
            gray = undistort_image(gray, K if resized_K is None else resized_K, dist)

        extracted = extract_sift(gray)
        if extracted is None:
            pts = np.empty((0, 2), dtype=np.float64)
            descs = np.empty((0, 128), dtype=np.float64)
        else:
            pts, descs = extracted
        images_data.append({
            'path': img_path,
            'gray': gray,
            'points': pts,
            'descriptors': descs,
            'color_img': img_resized if img_resized.ndim == 3 and img_resized.shape[2] >= 3 else None,
        })
        print(f"  [{i}] {os.path.basename(img_path)}: {len(pts)} keypoints")

    if resized_K is not None:
        K = resized_K

    print(f"\nStep 2/5: Feature matching...")

    for i in range(1, len(images_data)):
        matches = match_descriptors(
            images_data[i - 1]['descriptors'],
            images_data[i]['descriptors'],
            ratio=0.75,
        )
        images_data[i]['matches_with_prev'] = matches
        print(f"  Image {i-1} -> {i}: {len(matches)} matches")

    min_gap = max(3, len(images_data) // 15)   # ~7% of sequence (good baseline lower bound)
    max_gap = len(images_data) // 3             # at most 33% (avoid opposite side with no overlap)
    min_init_matches = 50
    best_match_idx = 1
    best_match_count = 0
    best_gap_match_idx = 1
    best_gap_match_count = 0

    for i in range(1, len(images_data)):
        m = match_descriptors(
            images_data[0]['descriptors'],
            images_data[i]['descriptors'],
            ratio=0.75,
        )
        gap = abs(i - 0)
        print(f"  Image 0 -> {i}: {len(m)} matches")
        if len(m) > best_match_count:
            best_match_count = len(m)
            best_match_idx = i
        if len(m) >= min_init_matches and min_gap <= gap <= max_gap and len(m) > best_gap_match_count:
            best_gap_match_count = len(m)
            best_gap_match_idx = i

    if best_gap_match_count > 0:
        best_match_idx = best_gap_match_idx
        print(f"  Selected init pair with gap={best_match_idx} ({best_gap_match_count} matches, baseline ~{best_match_idx/len(images_data)*360:.0f}°)")

    if best_match_idx != 1:
        images_data[1], images_data[best_match_idx] = images_data[best_match_idx], images_data[1]
        for i in range(1, len(images_data)):
            images_data[i].pop('matches_with_prev', None)
            images_data[i].pop('matches_with_prev2', None)
            images_data[i].pop('matches_with_prev3', None)
        for i in range(1, len(images_data)):
            images_data[i]['matches_with_prev'] = match_descriptors(
                images_data[i - 1]['descriptors'],
                images_data[i]['descriptors'],
                ratio=0.75,
            )
        print(f"  Reordered: init pair = 0 <-> {best_match_idx} ({best_match_count} matches)")

    if len(images_data) >= 2:
        matches_01 = match_descriptors(
            images_data[0]['descriptors'],
            images_data[1]['descriptors'],
            ratio=0.75,
        )
        images_data[1]['matches_with_0'] = matches_01
        print(f"  Image 0 -> 1 (init): {len(matches_01)} matches")

    for i in range(2, len(images_data)):
        matches = match_descriptors(
            images_data[i - 2]['descriptors'],
            images_data[i]['descriptors'],
            ratio=0.75,
        )
        images_data[i]['matches_with_prev2'] = matches
        print(f"  Image {i-2} -> {i} (prev2): {len(matches)} matches")

    for i in range(3, len(images_data)):
        matches = match_descriptors(
            images_data[i - 3]['descriptors'],
            images_data[i]['descriptors'],
            ratio=0.75,
        )
        images_data[i]['matches_with_prev3'] = matches
        print(f"  Image {i-3} -> {i} (prev3): {len(matches)} matches")

    print(f"\nStep 3/5: Incremental SfM reconstruction...")

    sfm_state = incremental_sfm(images_data, K, dist)

    n_cameras = len(sfm_state['cameras'])
    n_points = len(sfm_state['tracks'])
    print(f"  Registered cameras: {n_cameras}/{len(images_data)}")
    print(f"  Triangulated points: {n_points}")

    cameras_raw = sfm_state['cameras']
    tracks = sfm_state['tracks']
    registered_sorted = sorted(sfm_state['registered'])
    img_to_cam = {img_idx: cam_idx for cam_idx, img_idx in enumerate(registered_sorted)}

    cameras = []
    for cam_idx, img_idx in enumerate(registered_sorted):
        cam = cameras_raw[cam_idx]
        cameras.append({'R': cam['R'], 't': cam['t']})

    ba_track_ids = []
    ba_track_point3d = []
    ba_track_colors = []
    ba_observations = {}

    for track_id, track in tracks.items():
        if 'point3d' not in track or 'observations' not in track:
            continue
        obs_with_coords = {}
        for img_idx, feat_idx in track['observations'].items():
            if img_idx in img_to_cam:
                cam_idx = img_to_cam[img_idx]
                obs_with_coords[cam_idx] = images_data[img_idx]['points'][feat_idx].copy()
        if len(obs_with_coords) >= 2:
            ba_track_ids.append(track_id)
            ba_track_point3d.append(track['point3d'])
            ba_track_colors.append(track.get('color', (128, 128, 128)))
            ba_observations[track_id] = obs_with_coords

    if len(ba_track_point3d) == 0:
        print("  No 3D points with sufficient observations generated. Pipeline stopped.")
        return

    points3d = np.array(ba_track_point3d, dtype=np.float64)
    point_colors = np.array(ba_track_colors, dtype=np.uint8)

    print(f"\nStep 4/5: Bundle Adjustment...")

    if len(ba_observations) > 0:
        refined_cameras, refined_points = bundle_adjust(
            K, cameras, points3d, ba_observations,
            max_iterations=20, huber_delta=1.0
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
