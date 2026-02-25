# This code is part of:
#
#   CS4501-003: Computer Vision
#   University of Virginia
#   Instructor: Zezhou Cheng
#   Student: Nehemiah Kim

import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_laplace, maximum_filter


def detectBlobs(im, param=None):
    # Input:
    #   IM - input image
    #   PARAM - optional dictionary of hyperparameters for blob detection.
    #           Suggested keys:
    #           - sigma0: initial scale (recommended 2.0)
    #           - k: scale multiplication factor (recommended 1.25)
    #           - levels: number of scale levels (recommended 10)
    #           - threshold: minimum squared LoG response (start at 0.01)
    #           - topk: keep top scoring detections (e.g., 1000)
    #
    # Ouput:
    #   BLOBS - n x 4 array with blob in each row in (x, y, radius, score)
    #
    # Dummy - returns a blob at the center of the image
    if param is None:
        param = {}
    sigma0 = float(param.get("sigma0", 2.0))
    k = float(param.get("k", 1.25))
    levels = int(param.get("levels", 12))
    threshold = float(param.get("threshold", 0.01))
    topk = int(param.get("topk", 1000))

    # grayscale + float
    if im.ndim == 3 and im.shape[2] > 1:
        gray = rgb2gray(im)
    else:
        gray = im.copy()
    gray = gray.astype(np.float64)

    H, W = gray.shape[:2]

    #    response_l = (sigma^2 * LoG(I, sigma))^2
    sigmas = np.array([sigma0 * (k ** i) for i in range(levels)], dtype=np.float64)
    scale_space = np.zeros((levels, H, W), dtype=np.float64)

    for i, sigma in enumerate(sigmas):
        # gaussian_laplace computes ∇^2(Gσ * I)
        log_resp = gaussian_laplace(gray, sigma=sigma)
        norm_log = (sigma ** 2) * log_resp
        scale_space[i] = norm_log ** 2

    # maximum over a 3x3x3 neighborhood
    local_max = maximum_filter(scale_space, size=(3, 3, 3), mode="nearest")
    is_max = (scale_space == local_max)

    # Avoid selecting max-filter artifacts at the first/last scale
    is_max[0, :, :] = False
    is_max[-1, :, :] = False

    # Threshold
    is_max &= (scale_space >= threshold)

    # Extract detections
    s_idx, y_idx, x_idx = np.where(is_max)
    if len(s_idx) == 0:
        return np.zeros((0, 4), dtype=np.float64)

    scores = scale_space[s_idx, y_idx, x_idx]
    radii = np.sqrt(2.0) * sigmas[s_idx]  # common LoG blob radius mapping

    blobs = np.stack([x_idx.astype(np.float64),
                      y_idx.astype(np.float64),
                      radii.astype(np.float64),
                      scores.astype(np.float64)], axis=1)

    # Keep top-k by score
    order = np.argsort(-blobs[:, 3])
    blobs = blobs[order]
    if blobs.shape[0] > topk:
        blobs = blobs[:topk]

    return blobs
