import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


def _rodrigues(r):
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    Kx = np.array([[0, -k[2], k[1]],
                   [k[2], 0, -k[0]],
                   [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * (Kx @ Kx)


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


def bundle_adjust(K, cameras, tracks_points, observations,
                   max_iterations=50, huber_delta=1.0, lambda_init=1e-3):
    n_cams = len(cameras)
    n_pts = len(tracks_points)

    # --- pack parameter vector x = [rvec_0, t_0, ..., rvec_M, t_M, X_0, ..., X_N] ---
    x0 = np.zeros(6 * n_cams + 3 * n_pts)
    for i, cam in enumerate(cameras):
        x0[6*i:6*i+3] = _rotation_to_axis_angle(cam['R'])
        x0[6*i+3:6*i+6] = cam['t']
    x0[6*n_cams:] = np.asarray(tracks_points, dtype=np.float64).ravel()

    # --- precompute observation index arrays ---
    obs_cam = []
    obs_pt = []
    obs_uv = []
    for pt_idx, (track_id, obs_dict) in enumerate(observations.items()):
        for cam_idx, uv in obs_dict.items():
            if cam_idx < n_cams:
                obs_cam.append(cam_idx)
                obs_pt.append(pt_idx)
                obs_uv.append(np.asarray(uv, dtype=np.float64).flatten()[:2])
    obs_cam = np.array(obs_cam, dtype=np.int32)
    obs_pt = np.array(obs_pt, dtype=np.int32)
    obs_uv = np.array(obs_uv, dtype=np.float64)
    n_obs = len(obs_cam)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    def residual(params):
        cam_p = params[:6*n_cams].reshape(n_cams, 6)
        pts = params[6*n_cams:].reshape(n_pts, 3)
        r = np.empty(2 * n_obs)
        for k in range(n_obs):
            ci, pi = obs_cam[k], obs_pt[k]
            R = _rodrigues(cam_p[ci, :3])
            xc = R @ pts[pi] + cam_p[ci, 3:6]
            z_inv = 1.0 / xc[2]
            r[2*k]   = fx * xc[0] * z_inv + cx - obs_uv[k, 0]
            r[2*k+1] = fy * xc[1] * z_inv + cy - obs_uv[k, 1]
        return r

    def jacobian(params):
        cam_p = params[:6*n_cams].reshape(n_cams, 6)
        pts = params[6*n_cams:].reshape(n_pts, 3)

        J = lil_matrix((2 * n_obs, 6 * n_cams + 3 * n_pts))

        for k in range(n_obs):
            ci, pi = obs_cam[k], obs_pt[k]
            rvec = cam_p[ci, :3]
            t = cam_p[ci, 3:6]
            X = pts[pi]

            # Rodrigues + derivative w.r.t. rvec
            theta = np.linalg.norm(rvec)
            if theta < 1e-12:
                R = np.eye(3)
                # d(R@X)/drvec ≈ -[X]_×  (small-angle approximation)
                dR_dr = np.array([[0, X[2], -X[1]],
                                  [-X[2], 0, X[0]],
                                  [X[1], -X[0], 0]])
            else:
                k_vec = rvec / theta
                Kx = np.array([[0, -k_vec[2], k_vec[1]],
                               [k_vec[2], 0, -k_vec[0]],
                               [-k_vec[1], k_vec[0], 0]])
                R = np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * (Kx @ Kx)
                # Exact Rodrigues derivative: d(R@X)/drvec
                a = R @ X
                kX = np.cross(k_vec, X)
                dR_dr = (-np.sin(theta) * np.outer(a, k_vec)
                         + np.cos(theta) * np.outer(kX, k_vec)
                         + np.sin(theta) * (k_vec[:, None] @ kX[None, :] - np.eye(3) * (k_vec @ kX)))

            xc = R @ X + t
            z = xc[2]
            z2 = z * z

            # Pixel derivative w.r.t. camera coords
            du_dxc = np.array([fx / z, 0.0, -fx * xc[0] / z2])
            dv_dxc = np.array([0.0, fy / z, -fy * xc[1] / z2])

            # Chain to rvec, t, X
            du_dr = du_dxc @ dR_dr
            dv_dr = dv_dxc @ dR_dr
            du_dt = du_dxc
            dv_dt = dv_dxc
            du_dX = du_dxc @ R
            dv_dX = dv_dxc @ R

            row = 2 * k
            cam_col = 6 * ci
            pt_col = 6 * n_cams + 3 * pi

            J[row,   cam_col:cam_col+3] = du_dr
            J[row,   cam_col+3:cam_col+6] = du_dt
            J[row,   pt_col:pt_col+3] = du_dX
            J[row+1, cam_col:cam_col+3] = dv_dr
            J[row+1, cam_col+3:cam_col+6] = dv_dt
            J[row+1, pt_col:pt_col+3] = dv_dX

        return J.tocsr()

    result = least_squares(
        residual, x0, jacobian,
        method='trf',
        max_nfev=max_iterations,
        x_scale='jac',
        loss='huber' if huber_delta > 0 else 'linear',
        f_scale=huber_delta if huber_delta > 0 else 1.0,
        verbose=0,
    )

    # --- unpack ---
    cam_p_final = result.x[:6*n_cams].reshape(n_cams, 6)
    pts_final = result.x[6*n_cams:].reshape(n_pts, 3)

    cameras_out = []
    for i in range(n_cams):
        R = _rodrigues(cam_p_final[i, :3])
        cameras_out.append({'R': R, 't': cam_p_final[i, 3:6].copy()})

    return cameras_out, pts_final
