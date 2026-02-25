import numpy as np
from scipy.spatial.distance import cdist

# This code is part of:
#
#   CS4501-003: Computer Vision
#   University of Virginia
#   Instructor: Zezhou Cheng
#   Student: Nehemiah Kim
#

def computeMatches(f1, f2, ratio_thresh=0.8):
    """
    Match two sets of SIFT features f1 and f2 using SSD + Lowe ratio test.

    Input:
        f1: N x 128 array of SIFT features from image 1
        f2: M x 128 array of SIFT features from image 2
        ratio_thresh: Lowe ratio threshold (default 0.8)

    Output:
        matches: N array where matches[i] is the index in f2 that matches f1[i].
                 matches[i] = -1 if no match passes the ratio test.
    """
    if f1 is None or f2 is None or len(f1) == 0 or len(f2) == 0:
        return np.full((0,), -1, dtype=int)

    # Pairwise squared Euclidean distances (SSD)
    dist = cdist(f1, f2, metric="sqeuclidean")  # N x M

    # For each i, find best and second-best match in f2
    nn = np.argsort(dist, axis=1)              # N x M
    best = nn[:, 0]
    second = nn[:, 1] if dist.shape[1] > 1 else nn[:, 0]

    d1 = dist[np.arange(dist.shape[0]), best]
    d2 = dist[np.arange(dist.shape[0]), second]

    matches = np.full((dist.shape[0],), -1, dtype=int)

    # Ratio test: d1/d2 < ratio_thresh
    ok = (d2 > 1e-12) & ((d1 / d2) < ratio_thresh)
    matches[ok] = best[ok]

    return matches