# This code is part of:
#
#   CS4501-003: Computer Vision
#   University of Virginia
#   Instructor: Zezhou Cheng
#
import numpy as np


def alignChannels(img, max_shift):
    """Align the color channels of an image.
    
    The first channel (R) is fixed as reference. We compute the alignment 
    of the second (G) and third (B) channels with respect to the first.
    
    Args:
        img: np.array of size HxWx3 (three color channels).
        max_shift: np.array [max_shift_row, max_shift_col], the maximum 
                   shift to search in each direction.
    
    Returns:
        aligned_img: HxWx3 color image with aligned channels.
        pred_shift: 2x2 array where pred_shift[0] is the shift for channel 1 (G)
                    and pred_shift[1] is the shift for channel 2 (B).
                    Each shift is [shift_row, shift_col].
    
    Hints:
        - Use alignChannel() to find the best shift for each channel
        - Use shiftImage() to apply the shift
    """
    pred_shift = np.zeros((2, 2), dtype=int)

    # Reference is channel 0 (R)
    ref = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # Find best shifts for G and B to align to R
    shift_g = alignChannel(ref, g, max_shift)
    shift_b = alignChannel(ref, b, max_shift)

    # Apply shifts
    aligned = img.copy()
    aligned[:, :, 1] = shiftImage(g, shift_g)
    aligned[:, :, 2] = shiftImage(b, shift_b)

    pred_shift[0, :] = shift_g
    pred_shift[1, :] = shift_b

    return aligned, pred_shift


def alignChannel(ref_img, target_img, max_shift):
    """Find the best shift to align target_img to ref_img.
    
    Search over all possible shifts within [-max_shift, max_shift] range,
    compute a matching metric for each shift, and return the best shift.
    
    Args:
        ref_img: np.array of size HxW, the reference image (fixed).
        target_img: np.array of size HxW, the image to be aligned.
        max_shift: np.array [max_shift_row, max_shift_col].
    
    Returns:
        best_shift: np.array [shift_row, shift_col] that best aligns 
                    target_img to ref_img.
    
    Hints:
        - Use nested for loops to search over all shifts in the range
          [-max_shift[0], max_shift[0]] x [-max_shift[1], max_shift[1]]
        - For each shift, use shiftImage() to shift target_img
        - Compute a matching metric (SSD or Cosine Similarity) between 
          ref_img and the shifted target_img
        - Return the shift with the best score
    """
    best_shift = np.zeros(2)

    # Use SSD
    use_ssd = True

    # Central crop to reduce boundary artifacts 
    def central_crop(im, frac=0.10):
        H, W = im.shape
        dh = int(frac * H)
        dw = int(frac * W)
        return im[dh:H - dh, dw:W - dw]

    ref_c = central_crop(ref_img, frac=0.10)

    if use_ssd:
        best_score = np.inf
    else:
        best_score = -np.inf

    for dr in range(-max_shift[0], max_shift[0] + 1):
        for dc in range(-max_shift[1], max_shift[1] + 1):
            shifted = shiftImage(target_img, np.array([dr, dc]))
            shifted_c = central_crop(shifted, frac=0.10)

            if use_ssd:
                # SSD
                score = np.sum((ref_c - shifted_c) ** 2)
                if score < best_score:
                    best_score = score
                    best_shift = np.array([dr, dc])
            else:
                # Cosine similarity
                a = ref_c.reshape(-1)
                b = shifted_c.reshape(-1)
                denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                score = float(np.dot(a, b) / denom)
                if score > best_score:
                    best_score = score
                    best_shift = np.array([dr, dc])

    return best_shift.astype(int)


def shiftImage(img, shift):
    """Shift an image by the given amount.
    
    Args:
        img: np.array of size HxW.
        shift: np.array [shift_row, shift_col], the amount to shift.
               Positive values shift down/right, negative values shift up/left.
    
    Returns:
        shifted_img: HxW image shifted by the specified amount.
    
    Hints:
        - You can use np.roll() for circular shift, which works well for 
          toy examples
        - For real images, consider using np.pad() with 'edge' mode to 
          handle boundaries
    """
    return np.roll(img, [int(shift[0]), int(shift[1])], axis=[0, 1])