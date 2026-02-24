# This code is part of:
#
#   CS4501-003: Computer Vision
#   University of Virginia
#   Instructor: Zezhou Cheng
#

import numpy as np


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
    blobs = np.array([[im.shape[1]/2, im.shape[0]/2, 100, 1.0]])
    return np.array(blobs)
