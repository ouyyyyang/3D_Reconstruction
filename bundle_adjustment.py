import numpy as np


def _rodrigues(r):
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3), np.eye(3)
    k = r / theta
    Kx = np.array([[0, -k[2], k[1]],
                   [k[2], 0, -k[0]],
                   [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * (Kx @ Kx)

    a = np.eye(3)
    b = Kx
    c = (1 - np.cos(theta)) / theta * Kx @ Kx
    dR_dr = a + np.sin(theta) * b + c
    return R, dR_dr


def _project(K, rvec, t, X):
    R, _ = _rodrigues(rvec)
    x = R @ X + t
    u = K[0, 0] * x[0] / x[2] + K[0, 2]
    v = K[1, 1] * x[1] / x[2] + K[1, 2]
    return np.array([u, v]), x


def _compute_jacobian(K, rvec, t, X, eps=1e-6):
    u0, _ = _project(K, rvec, t, X)
    J_camera = np.zeros((2, 6))
    for i in range(6):
        delta = np.zeros(6)
        delta[i] = eps
        rvec_p = rvec + delta[:3]
        t_p = t + delta[3:6]
        up, _ = _project(K, rvec_p, t_p, X)
        J_camera[:, i] = (up - u0) / eps

    J_point = np.zeros((2, 3))
    for i in range(3):
        delta = np.zeros(3)
        delta[i] = eps
        up, _ = _project(K, rvec, t, X + delta)
        J_point[:, i] = (up - u0) / eps

    return J_camera, J_point


def _quaternion_to_rotation(q):
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])
    return R


def _axis_angle_to_quaternion(rvec):
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rvec / theta
    half_theta = theta / 2.0
    w = np.cos(half_theta)
    s = np.sin(half_theta)
    return np.array([w, s*axis[0], s*axis[1], s*axis[2]])


def _assemble_system(cam_params, points3d, K, observations, huber_delta=1.0):
    n_cams = len(cam_params)
    n_pts = len(points3d)

    num_obs = sum(len(obs) for obs in observations.values())

    cam_block_size = 6
    point_block_size = 3

    J_list = []
    r_list = []
    obs_info = []

    for pt_idx, (track_id, obs_dict) in enumerate(observations.items()):
        for cam_idx, feat_idx in obs_dict.items():
            u_obs = np.array(feat_idx) if isinstance(feat_idx, (list, tuple, np.ndarray)) else feat_idx

    obs_map = []
    track_id_to_idx = {}
    for pt_idx, track_id in enumerate(observations.keys()):
        track_id_to_idx[track_id] = pt_idx

    for track_id, obs_dict in observations.items():
        pt_idx = track_id_to_idx[track_id]
        X = points3d[pt_idx]
        for cam_idx, feat_idx in obs_dict.items():
            obs_map.append({
                'cam_idx': cam_idx,
                'pt_idx': pt_idx,
                'obs': feat_idx,
                'global_cam_idx': cam_idx,
            })

    U = np.zeros((6 * n_cams, 6 * n_cams))
    V_diag = [np.zeros((3, 3)) for _ in range(n_pts)]
    W_list = [{} for _ in range(n_pts)]
    g_c = np.zeros(6 * n_cams)
    g_p = np.zeros(3 * n_pts)

    total_error = 0.0
    num_valid = 0

    for track_id, obs_dict in observations.items():
        if track_id in track_id_to_idx:
            pt_idx = track_id_to_idx[track_id]
        else:
            continue

        for cam_idx, feat_idx in obs_dict.items():
            if cam_idx >= n_cams:
                continue

            rvec = cam_params[cam_idx][:3]
            tvec = cam_params[cam_idx][3:6]

            u_proj, x_cam = _project(K, rvec, tvec, points3d[pt_idx])

            J_cam, J_pt = _compute_jacobian(K, rvec, tvec, points3d[pt_idx])

            u_obs_pt = feat_idx[:2] if isinstance(feat_idx, (tuple, list, np.ndarray)) else feat_idx
            residual = u_obs_pt - u_proj

            err = np.linalg.norm(residual)
            rho = 1.0
            if huber_delta > 0 and err > huber_delta:
                rho = np.sqrt(huber_delta / err)
            total_error += err * err * rho

            cam_start = 6 * cam_idx
            pt_start = 3 * pt_idx

            g_c[cam_start:cam_start + 6] += rho * J_cam.T @ residual
            g_p[pt_start:pt_start + 3] += rho * J_pt.T @ residual

            U[cam_start:cam_start + 6, cam_start:cam_start + 6] += rho * (J_cam.T @ J_cam)
            V_diag[pt_idx] += rho * (J_pt.T @ J_pt)

            W = rho * (J_cam.T @ J_pt)
            if pt_idx in W_list:
                if cam_idx in W_list[pt_idx]:
                    W_list[pt_idx][cam_idx] += W
                else:
                    W_list[pt_idx][cam_idx] = W.copy()

            num_valid += 1

    return U, V_diag, W_list, g_c, g_p, total_error, num_valid


def bundle_adjust(K, cameras, tracks_points, observations,
                  max_iterations=20, huber_delta=1.0, lambda_init=1e-3):
    n_cams = len(cameras)
    n_pts = len(tracks_points)

    cam_params = np.zeros((n_cams, 6))
    for i, cam in enumerate(cameras):
        rvec = _rotation_to_axis_angle(cam['R'])
        cam_params[i, :3] = rvec
        cam_params[i, 3:6] = cam['t']

    points3d = np.array(tracks_points, dtype=np.float64)

    lambda_ = lambda_init
    prev_error = float('inf')

    for iteration in range(max_iterations):
        U, V_diag, W_list, g_c, g_p, total_error, num_valid = _assemble_system(
            cam_params, points3d, K, observations, huber_delta)

        if num_valid == 0:
            break

        lambda_I_cam = lambda_ * np.eye(6 * n_cams)
        U_aug = U + lambda_I_cam

        S = np.zeros((6 * n_cams, 6 * n_cams))
        s_vec = np.zeros(6 * n_cams)

        for pt_idx in range(n_pts):
            V_aug = V_diag[pt_idx] + lambda_ * np.eye(3)
            try:
                V_inv = np.linalg.inv(V_aug)
            except np.linalg.LinAlgError:
                V_inv = np.linalg.pinv(V_aug)

            for cam_idx, W_mat in W_list[pt_idx].items():
                cam_start = 6 * cam_idx
                pt_start = 3 * pt_idx

                WVinv = W_mat @ V_inv
                S[cam_start:cam_start + 6, cam_start:cam_start + 6] += WVinv @ W_mat.T
                s_vec[cam_start:cam_start + 6] += WVinv @ g_p[pt_start:pt_start + 3]

                for cam_idx2, W2_mat in W_list[pt_idx].items():
                    if cam_idx2 > cam_idx:
                        cam_start2 = 6 * cam_idx2
                        SW = WVinv @ W2_mat.T
                        S[cam_start:cam_start + 6, cam_start2:cam_start2 + 6] += SW
                        S[cam_start2:cam_start2 + 6, cam_start:cam_start + 6] += SW.T

        A = U_aug - S
        b = g_c - s_vec

        try:
            delta_c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            delta_c = np.linalg.lstsq(A, b, rcond=None)[0]

        delta_p = np.zeros(3 * n_pts)
        for pt_idx in range(n_pts):
            pt_start = 3 * pt_idx
            V_aug = V_diag[pt_idx] + lambda_ * np.eye(3)
            rhs = g_p[pt_start:pt_start + 3]
            for cam_idx, W_mat in W_list[pt_idx].items():
                delta_block = delta_c[6 * cam_idx:6 * cam_idx + 6]
                rhs -= W_mat.T @ delta_block
            try:
                delta_p[pt_start:pt_start + 3] = np.linalg.solve(V_aug, rhs)
            except np.linalg.LinAlgError:
                delta_p[pt_start:pt_start + 3] = np.linalg.lstsq(V_aug, rhs, rcond=None)[0]

        current_error = total_error

        cam_params_new = cam_params + delta_c.reshape(n_cams, 6)
        points3d_new = points3d + delta_p.reshape(n_pts, 3)

        _, _, _, _, _, new_error, _ = _assemble_system(
            cam_params_new, points3d_new, K, observations, huber_delta)

        if new_error < current_error:
            cam_params = cam_params_new
            points3d = points3d_new
            lambda_ = max(lambda_ * 0.5, 1e-12)
            if current_error > 0 and (current_error - new_error) / current_error < 1e-8:
                break
        else:
            lambda_ = min(lambda_ * 2.0, 1e6)

    cameras_out = []
    for i in range(n_cams):
        rvec = cam_params[i, :3]
        R = _rodrigues(rvec)[0]
        t = cam_params[i, 3:6]
        cameras_out.append({'R': R, 't': t})

    return cameras_out, points3d


def _rotation_to_axis_angle(R):
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-12:
        return np.zeros(3)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2 * np.sin(theta))
    return axis * theta
