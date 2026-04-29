import numpy as np


def decompose_essential(E):
    U, _, Vt = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    candidates = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t),
    ]
    return candidates


def _triangulate_one(P1, P2, pt1, pt2):
    A = np.zeros((4, 4), dtype=np.float64)
    A[0] = pt1[0] * P1[2] - P1[0]
    A[1] = pt1[1] * P1[2] - P1[1]
    A[2] = pt2[0] * P2[2] - P2[0]
    A[3] = pt2[1] * P2[2] - P2[1]

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]
    return X[:3]


def select_pose(candidates, pts1, pts2, K1, K2):
    P1 = K1 @ np.column_stack([np.eye(3), np.zeros(3)])

    best_count = 0
    best_R = None
    best_t = None
    best_points = None

    for R, t in candidates:
        P2 = K2 @ np.column_stack([R, t])

        count_pos = 0
        points_3d = []
        for pt1, pt2 in zip(pts1, pts2):
            X = _triangulate_one(P1, P2, pt1, pt2)
            x1 = P1 @ np.append(X, 1.0)
            x2 = P2 @ np.append(X, 1.0)
            if x1[2] > 0 and x2[2] > 0:
                count_pos += 1
                points_3d.append(X)

        if count_pos > best_count:
            best_count = count_pos
            best_R = R
            best_t = t
            best_points = np.array(points_3d)

    return best_R, best_t, best_points
