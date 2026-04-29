import numpy as np


def triangulate_dlt(P1, P2, pt1, pt2):
    if len(pt1) == 0:
        return np.empty((0, 3))

    n = len(pt1)
    points_3d = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        A = np.zeros((4, 4), dtype=np.float64)
        A[0] = pt1[i][0] * P1[2] - P1[0]
        A[1] = pt1[i][1] * P1[2] - P1[1]
        A[2] = pt2[i][0] * P2[2] - P2[0]
        A[3] = pt2[i][1] * P2[2] - P2[1]
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]
        points_3d[i] = X[:3]
    return points_3d


def _project(P, X):
    x = P @ np.append(X, 1.0)
    if x[2] < 1e-10:
        x[2] = 1e-10
    return x[:2] / x[2]


def reprojection_error(P, X, pt_obs):
    proj = _project(P, X)
    return np.linalg.norm(proj - pt_obs)


def filter_by_error(points_3d, P1, P2, pts1, pts2, max_error=4.0):
    mask = np.ones(len(points_3d), dtype=bool)
    for i in range(len(points_3d)):
        err1 = reprojection_error(P1, points_3d[i], pts1[i])
        err2 = reprojection_error(P2, points_3d[i], pts2[i])
        if err1 > max_error or err2 > max_error:
            mask[i] = False
    return mask
