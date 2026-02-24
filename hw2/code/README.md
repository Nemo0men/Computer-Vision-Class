# Homework 2 Coding Notes

## Blob Detection Hints (`detectBlobs`)

Previously, there are many question regarding Blob detection. Therefore, please use these hints to make your implementation stable and easier to debug.

### 1) Recommended starting parameters

- `initial scale (sigma0)`: `2.0`
- `scale factor (k)`: `1.25`
- `number of levels`: `10`
- `response threshold`: start from `0.01` and tune
- keep top scoring detections (for example top `1000`)

These are good defaults before any tuning.

### 2) Suggested implementation order

1. Convert image to grayscale and `float` in `[0, 1]`.
2. Build a Laplacian scale space across levels.
3. Use scale-normalized Laplacian response (`sigma^2 * LoG`) and square it.
4. Run 3D non-maximum suppression over `(x, y, scale)` (3x3x3 neighborhood).
5. Keep points above threshold.
6. Convert selected scale to display radius.
7. Return blobs as `N x 4`: `(x, y, radius, score)`.

### 3) Common pitfalls

- Forgetting grayscale + float conversion.
- Doing NMS only in 2D (must include scale axis).
- Using inconsistent coordinate order (`x, y` vs `row, col`).
- Returning too many noisy points because threshold is too low.
- Crashing on empty detections (always return a valid `N x 4` array).

### 4) Quick sanity checks

- The output shape should be `N x 4`.
- Scores should be non-negative after squaring response.
- Increasing threshold should reduce blob count.
- Different images should not produce exactly identical detections.

