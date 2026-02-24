# This code is part of:
#
#   CS4501-003: Computer Vision
#   University of Virginia
#   Instructor: Zezhou Cheng
#

import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import imread
from skimage.color import rgb2gray
from skimage.feature import corner_orientations
from skimage.feature import hog


def compute_sift(I, circles, enlarge_factor=1.5):
# I - image
# circles - Nx3 array where N is the number of circles, where the
#    first column is the x-coordinate, the second column is the y-coordinate,
#    and the third column is the radius
# enlarge_factor is by how much to enarge the radius of the circle before
#    computing the descriptor (a factor of 1.5 or larger is usually necessary
#    for best performance)
# The output is an Nx128 array of SIFT descriptors


    sift = cv2.SIFT_create()


    angle_bins = 36
    I = rgb2gray(I)
    hist = hog(I, orientations=angle_bins, pixels_per_cell=(10, 10),
            cells_per_block=(1,1), feature_vector=False, block_norm='L2-Hys')


    xcoord = np.floor(circles[:,0]/10.0).astype(int)
    ycoord = np.floor(circles[:,1]/10.0).astype(int)
    
    # Clip coordinates to ensure they're within bounds
    xcoord = np.clip(xcoord, 0, hist.shape[1]-1)
    ycoord = np.clip(ycoord, 0, hist.shape[0]-1)

    circ_hist = hist[ycoord, xcoord, 0, 0, :]
    angles = np.rad2deg(np.argmax(circ_hist, axis=1) * 2*np.pi/angle_bins)


    img_gray = (I*255.0).astype('uint8')


    kpts = []
    for i in range(angles.shape[0]):
        kpts.append(cv2.KeyPoint(circles[i, 0], circles[i, 1], 
            size=enlarge_factor*circles[i, 2],
            angle=angles[i]))


    _, des = sift.compute(img_gray, kpts)
    return des






if __name__ == '__main__':
    sift = cv2.SIFT_create()
    img1 = cv2.imread("../data/stitching/hill_1.jpg")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread("../data/stitching/hill_2.jpg")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1 = sift.detect(img1, None)
    kp2 = sift.detect(img2, None)


    I1 = imread("../data/stitching/hill_1.jpg")
    I2 = imread("../data/stitching/hill_2.jpg")
    circles1 = [[kp1[i].pt[0], kp1[i].pt[1], kp1[i].size] for i in range(len(kp1))]
    circles1 = np.array(circles1)
    circles2 = [[kp2[i].pt[0], kp2[i].pt[1], kp2[i].size] for i in range(len(kp2))]
    circles2 = np.array(circles2)


    desc1 = compute_sift(I1, circles1)
    desc2 = compute_sift(I2, circles2)


    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(desc1,desc2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50], None, flags=2)


    plt.imshow(img3),plt.show()

    plt.savefig('sift_matches.png')
