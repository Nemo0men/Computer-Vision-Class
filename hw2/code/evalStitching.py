# This code is part of:
#
#   CS4501-003: Computer Vision
#   University of Virginia
#   Instructor: Zezhou Cheng
#

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # prevents pop-up windows; enables saving only
import matplotlib.pyplot as plt

from utils import imread, showMatches
from detectBlobs import detectBlobs
from computeSift import compute_sift
from computeMatches import computeMatches
from ransac import ransac
from mergeImages import mergeImages

# Input directory
dataDir = os.path.join('..', 'data', 'stitching')

# Output directory
outDir = os.path.join('..', 'output', 'stitching')
os.makedirs(outDir, exist_ok=True)

# All test examples (expects files like book_1.jpg, book_2.jpg, etc.)
testExamples = ['book', 'hill', 'house', 'kitchen', 'park', 'pier', 'roof', 'table']

# Save affine matrices here
affine_path = os.path.join(outDir, "affine_transforms.txt")
with open(affine_path, "w") as f:
    f.write("example,m11,m12,t1,m21,m22,t2\n")

for ex in testExamples:
    imageName1 = f'{ex}_1.jpg'
    imageName2 = f'{ex}_2.jpg'

    path1 = os.path.join(dataDir, imageName1)
    path2 = os.path.join(dataDir, imageName2)

    print(f"\n=== Processing {ex}: {imageName1} + {imageName2} ===")

    im1 = imread(path1)
    im2 = imread(path2)

    # 1) Detect keypoints (blobs)
    blobs1 = detectBlobs(im1)
    blobs2 = detectBlobs(im2)

    if blobs1.shape[0] == 0 or blobs2.shape[0] == 0:
        print(f"[WARN] No blobs detected for {ex}. Skipping.")
        continue

    # 2) Compute SIFT descriptors on blobs
    sift1 = compute_sift(im1, blobs1[:, 0:4])
    sift2 = compute_sift(im2, blobs2[:, 0:4])

    if sift1 is None or sift2 is None or len(sift1) == 0 or len(sift2) == 0:
        print(f"[WARN] SIFT failed/empty for {ex}. Skipping.")
        continue

    # 3) Compute raw matches
    matches = computeMatches(sift1, sift2)

    # Save raw matches visualization
    showMatches(im1, im2, blobs1, blobs2, matches)
    plt.title(f"Raw matches: {ex}")
    plt.savefig(os.path.join(outDir, f"{ex}_matches_raw.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4) RANSAC -> inliers + affine transform (2x3)
    inliers, transf = ransac(matches, blobs1, blobs2)

    # Build inlier-only matches array for visualization
    goodMatches = np.full_like(matches, -1)
    if len(inliers) > 0:
        goodMatches[inliers] = matches[inliers]

    # Save inlier matches visualization
    showMatches(im1, im2, blobs1, blobs2, goodMatches)
    plt.title(f"RANSAC inliers: {ex} (n={len(inliers)})")
    plt.savefig(os.path.join(outDir, f"{ex}_matches_inliers.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Save affine matrix to file
    m11, m12, t1 = transf[0, 0], transf[0, 1], transf[0, 2]
    m21, m22, t2 = transf[1, 0], transf[1, 1], transf[1, 2]
    with open(affine_path, "a") as f:
        f.write(f"{ex},{m11:.6f},{m12:.6f},{t1:.6f},{m21:.6f},{m22:.6f},{t2:.6f}\n")

    print("Affine (2x3):")
    print(transf)

    # 5) Stitch + save
    stitchIm = mergeImages(im1, im2, transf)
    plt.figure()
    plt.imshow(stitchIm)
    plt.title(f"Stitched: {ex}")
    plt.axis("off")
    plt.savefig(os.path.join(outDir, f"{ex}_stitched.png"), dpi=300, bbox_inches="tight")
    plt.close()

print(f"\nDone. Outputs saved to: {outDir}")
print(f"Affine transforms saved to: {affine_path}")