import numpy as np
import os


def export_ply(points, colors, filename, normals=None):
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

    if colors is not None and len(colors) == len(points):
        if colors.dtype != np.uint8:
            colors = np.clip(colors, 0, 255).astype(np.uint8)
    else:
        colors = np.ones((len(points), 3), dtype=np.uint8) * 200

    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        if normals is not None:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        f.write("end_header\n")
        for i in range(len(points)):
            r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
            if normals is not None:
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                        f"{r} {g} {b} "
                        f"{normals[i, 0]:.6f} {normals[i, 1]:.6f} {normals[i, 2]:.6f}\n")
            else:
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                        f"{r} {g} {b}\n")

    return filename


def _create_camera_frustum(R, t, color=(1, 0, 0), scale=0.1, K=None):
    camera_points = np.array([
        [0, 0, 0],
        [1, 1, 2],
        [-1, 1, 2],
        [-1, -1, 2],
        [1, -1, 2],
    ]) * scale

    if K is not None:
        fx = K[0, 0]
        aspect = K[1, 1] / K[0, 0]
        camera_points = np.array([
            [0, 0, 0],
            [1 * scale, aspect * scale, 2 * scale * fx / K[0, 0]],
            [-1 * scale, aspect * scale, 2 * scale * fx / K[0, 0]],
            [-1 * scale, -aspect * scale, 2 * scale * fx / K[0, 0]],
            [1 * scale, -aspect * scale, 2 * scale * fx / K[0, 0]],
        ])

    camera_points = (R @ camera_points.T).T + t

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ]
    return camera_points, lines, color


def visualize_open3d(points, cameras=None, point_colors=None, window_name="SfM Reconstruction"):
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not installed. Install with: pip install open3d")
        return

    pcd = o3d.geometry.PointCloud()

    if point_colors is not None:
        if point_colors.dtype != np.float64:
            point_colors = point_colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(point_colors[:, :3])
    else:
        colors = np.ones((len(points), 3)) * 0.7
        pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd.points = o3d.utility.Vector3dVector(points)

    geometries = [pcd]

    colors_rgb = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                  (1, 1, 0), (1, 0, 1), (0, 1, 1),
                  (0.5, 0.2, 0.8), (0.8, 0.5, 0.2)]

    if cameras is not None:
        for i, cam in enumerate(cameras):
            if 'R' in cam and 't' in cam:
                R = cam['R']
                t = cam['t']
                color = colors_rgb[i % len(colors_rgb)]
                frustum_pts, lines, _ = _create_camera_frustum(R, t, color=color)

                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(frustum_pts)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                color_arr = np.array([color for _ in range(len(lines))])
                line_set.colors = o3d.utility.Vector3dVector(color_arr)
                geometries.append(line_set)

    o3d.visualization.draw_geometries(geometries, window_name=window_name)
