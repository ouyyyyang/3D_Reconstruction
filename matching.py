import numpy as np


def match_descriptors(desc1, desc2, ratio=0.75, cross_check=True):
    if len(desc1) == 0 or len(desc2) == 0:
        return np.empty((0, 2), dtype=np.int32)

    matches_12 = _match_forward(desc1, desc2, ratio)
    if not cross_check:
        return matches_12

    matches_21 = _match_forward(desc2, desc1, ratio)

    cross = {}
    for j, i in matches_21:
        cross[j] = i

    verified = []
    for i, j in matches_12:
        if j in cross and cross[j] == i:
            verified.append((i, j))

    return np.array(verified, dtype=np.int32)


def _match_forward(desc1, desc2, ratio):
    if len(desc1) == 0 or len(desc2) == 0:
        return np.empty((0, 2), dtype=np.int32)

    dists = _pairwise_l2(desc1, desc2)
    matches = []
    for i in range(len(desc1)):
        row = dists[i]
        idx = np.argpartition(row, min(2, len(row) - 1))
        best = idx[0]
        best_dist = row[best]
        if len(row) >= 2:
            second = idx[1]
            second_dist = row[second]
            if best_dist < ratio * second_dist:
                matches.append((i, best))
        else:
            matches.append((i, best))
    return np.array(matches, dtype=np.int32)


def _pairwise_l2(a, b):
    a_sq = np.sum(a ** 2, axis=1, keepdims=True)
    b_sq = np.sum(b ** 2, axis=1, keepdims=True)
    ab = np.dot(a, b.T)
    dists = a_sq - 2.0 * ab + b_sq.T
    dists = np.maximum(dists, 0)
    return np.sqrt(dists)
