import numpy as np
from scipy.spatial.distance import cdist

# This code is part of:
#
#   CS4501-003: Computer Vision
#   University of Virginia
#   Instructor: Zezhou Cheng
#

def computeMatches(f1, f2):
    """
    Match two sets of SIFT features f1 and f2.
    
    Input:
        f1: N x 128 array of SIFT features from image 1
        f2: M x 128 array of SIFT features from image 2
    
    Output:
        matches: N x 1 array where matches[i] is the index in f2 that matches the i-th feature in f1.
                 matches[i] = -1 if no match is found.
    """
    # implement this

