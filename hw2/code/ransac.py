import numpy as np
import cv2

# This code is part of:
#
#   CS4501-003: Computer Vision
#   University of Virginia
#   Instructor: Zezhou Cheng
#   Student: Nehemiah Kim
#

def ransac(matches, blobs1, blobs2, num_iters=2000, inlier_thresh=3.0):

    # Collect valid correspondences
    idx1 = np.where(matches >= 0)[0]
    if idx1.size < 3:
        # Not enough matches to estimate affine
        return np.array([], dtype=int), np.array([[1, 0, 0],
                                                  [0, 1, 0]], dtype=float)

    idx2 = matches[idx1].astype(int)

    # Points: blobs are stored as (x, y, r, score)
    pts1 = blobs1[idx1, 0:2].astype(np.float32)  # image1
    pts2 = blobs2[idx2, 0:2].astype(np.float32)  # image2

    best_inliers = np.array([], dtype=int)
    best_M = None

    N = pts1.shape[0]
    rng = np.random.default_rng()

    for _ in range(num_iters):
        # sample 3 correspondences
        sample = rng.choice(N, size=3, replace=False)
        p1_s = pts1[sample]
        p2_s = pts2[sample]

        # estimate affine: map p2 -> p1
        M = cv2.getAffineTransform(p2_s, p1_s)  # 2x3

        # apply transform to all pts2
        pts2_h = np.hstack([pts2, np.ones((N, 1), dtype=np.float32)])  # N x 3
        pred1 = (M @ pts2_h.T).T  # N x 2

        # compute reprojection error
        err = np.linalg.norm(pred1 - pts1, axis=1)
        inl = np.where(err < inlier_thresh)[0]

        if inl.size > best_inliers.size:
            best_inliers = inl
            best_M = M

    if best_M is None or best_inliers.size < 3:
        return np.array([], dtype=int), np.array([[1, 0, 0],
                                                  [0, 1, 0]], dtype=float)

    # Refit affine using all inliers for a better estimate
    inlier_pts1 = pts1[best_inliers]
    inlier_pts2 = pts2[best_inliers]

    # estimateAffine2D returns (M, inlierMask); use least-squares refinement
    M_refit, _ = cv2.estimateAffine2D(inlier_pts2, inlier_pts1)
    if M_refit is None:
        M_refit = best_M

    # Convert inliers back to indices in the original matches array (blobs1 indices)
    inliers_global = idx1[best_inliers].astype(int)

    return inliers_global, M_refit.astype(float)