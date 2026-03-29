import cv2
import numpy as np
import tifffile as tiff
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WIDTH = 1600
HEIGHT = 1400

MIRA_RAW_PATH = "Vijay_0315_leaves_pic1.raw"
MIRA_BLACK_PATH = "dark.raw"
MIRA_WHITE_PATH = "white.raw"
MIRA_PNG_PATH = "03_color_corrected_Bilinear_Debayering.png"

MAPIR_ALIGNED_PATH = "Vijay_0315_leaves_tif_aligned_to_mira.tif"
MAPIR_MASK_PATH = "Vijay_0315_leaves_tif_aligned_mask.png"

MIRA_NDVI_ARRAY_OUT = "03_color_corrected_Bilinear_Debayering_ndvi.npy"
MIRA_NDVI_HEATMAP_OUT = "03_color_corrected_Bilinear_Debayering_ndvi_heatmap.png"
MAPIR_NDVI_ARRAY_OUT = "Vijay_0315_leaves_tif_aligned_to_mira_ndvi.npy"
MAPIR_NDVI_HEATMAP_OUT = "Vijay_0315_leaves_tif_aligned_to_mira_ndvi_heatmap.png"

ARUCO_DICT = cv2.aruco.DICT_4X4_1000
MARKER_ID = 830

SPECTRAL_MATRIX = np.array(
    [
        [3.8798146, -1.0125873, 0.03123079, -3.759391],
        [-4.4472, 4.988873, -2.482766, 5.090941],
        [1.2792171, -3.7578082, 5.13936, -1.2770654],
        [0.0, 0.0, 0.0, 2.0],
    ],
    dtype=np.float32,
)


def read_raw(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint16)
    expected = WIDTH * HEIGHT
    if data.size != expected:
        raise ValueError(f"{path}: expected {expected} pixels, got {data.size}")
    return data.reshape((HEIGHT, WIDTH))


def blockflip_4x4(image: np.ndarray) -> np.ndarray:
    reshaped = image.reshape(HEIGHT // 4, 4, WIDTH // 4, 4)
    flipped = reshaped[:, :, :, ::-1]
    return flipped.reshape(HEIGHT, WIDTH)


def calibrate_bayer(raw: np.ndarray, black: np.ndarray, white: np.ndarray) -> np.ndarray:
    black_f = blockflip_4x4(black).astype(np.float32)
    white_f = blockflip_4x4(white).astype(np.float32)
    raw_f = raw.astype(np.float32)

    denom = white_f - black_f
    denom[denom == 0] = 1.0
    calibrated = (raw_f - black_f) / denom
    return np.clip(calibrated, 0.0, 1.0)


def debayer_mira(calibrated: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flipped = blockflip_4x4(calibrated)

    r = np.zeros_like(flipped, dtype=np.float32)
    g = np.zeros_like(flipped, dtype=np.float32)
    b = np.zeros_like(flipped, dtype=np.float32)
    ir = np.zeros_like(flipped, dtype=np.float32)

    b[0::4, 0::4] = flipped[0::4, 0::4]
    g[0::4, 1::4] = flipped[0::4, 1::4]
    r[0::4, 2::4] = flipped[0::4, 2::4]
    g[0::4, 3::4] = flipped[0::4, 3::4]

    g[1::4, 0::4] = flipped[1::4, 0::4]
    ir[1::4, 1::4] = flipped[1::4, 1::4]
    g[1::4, 2::4] = flipped[1::4, 2::4]
    ir[1::4, 3::4] = flipped[1::4, 3::4]

    r[2::4, 0::4] = flipped[2::4, 0::4]
    g[2::4, 1::4] = flipped[2::4, 1::4]
    b[2::4, 2::4] = flipped[2::4, 2::4]
    g[2::4, 3::4] = flipped[2::4, 3::4]

    g[3::4, 0::4] = flipped[3::4, 0::4]
    ir[3::4, 1::4] = flipped[3::4, 1::4]
    g[3::4, 2::4] = flipped[3::4, 2::4]
    ir[3::4, 3::4] = flipped[3::4, 3::4]

    return r, g, b, ir


def fill_sparse_channel(channel: np.ndarray, iterations: int = 2) -> np.ndarray:
    result = channel.astype(np.float32).copy()
    for _ in range(iterations):
        zeros = result == 0
        up = np.roll(result, 1, axis=0)
        down = np.roll(result, -1, axis=0)
        left = np.roll(result, 1, axis=1)
        right = np.roll(result, -1, axis=1)

        neighbor_sum = up + down + left + right
        neighbor_count = (up > 0).astype(np.float32)
        neighbor_count += (down > 0).astype(np.float32)
        neighbor_count += (left > 0).astype(np.float32)
        neighbor_count += (right > 0).astype(np.float32)

        fill_mask = zeros & (neighbor_count > 0)
        result[fill_mask] = neighbor_sum[fill_mask] / neighbor_count[fill_mask]

        result[0, :] = channel[0, :]
        result[-1, :] = channel[-1, :]
        result[:, 0] = channel[:, 0]
        result[:, -1] = channel[:, -1]

    return result


def apply_spectral_matrix(r: np.ndarray, g: np.ndarray, b: np.ndarray, ir: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.stack([r, g, b, ir], axis=-1)
    corrected = stacked @ SPECTRAL_MATRIX.T
    corrected = np.clip(corrected, 0.0, 1.0)
    return tuple(corrected[:, :, idx] for idx in range(4))


def compute_ndvi(ir: np.ndarray, r: np.ndarray) -> np.ndarray:
    denom = ir.astype(np.float32) + r.astype(np.float32)
    ndvi = np.divide(ir - r, denom, out=np.zeros_like(ir, dtype=np.float32), where=denom != 0)
    return np.clip(ndvi, -1.0, 1.0)


def normalize_to_u8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image_min = float(image.min())
    image_span = max(float(image.max() - image_min), 1e-6)
    return ((image - image_min) * 255.0 / image_span).astype(np.uint8)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_marker_corners(gray: np.ndarray) -> np.ndarray:
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
        cv2.aruco.DetectorParameters(),
    )
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        raise RuntimeError("No ArUco markers detected")

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        if marker_id == MARKER_ID:
            return marker_corners.reshape(4, 2).astype(np.float32)

    raise RuntimeError(f"Marker {MARKER_ID} not found")


def order_corners(corners: np.ndarray) -> np.ndarray:
    sums = corners.sum(axis=1)
    diffs = np.diff(corners, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = corners[np.argmin(sums)]
    ordered[2] = corners[np.argmax(sums)]
    ordered[1] = corners[np.argmin(diffs)]
    ordered[3] = corners[np.argmax(diffs)]
    return ordered


def save_heatmap(ndvi: np.ndarray, out_path: str, title: str, mask: np.ndarray | None = None) -> None:
    plot_ndvi = ndvi.copy()
    if mask is not None:
        plot_ndvi = np.where(mask, plot_ndvi, np.nan)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(plot_ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NDVI")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    mira_raw = read_raw(MIRA_RAW_PATH)
    mira_black = read_raw(MIRA_BLACK_PATH)
    mira_white = read_raw(MIRA_WHITE_PATH)

    calibrated = calibrate_bayer(mira_raw, mira_black, mira_white)
    r, g, b, ir = debayer_mira(calibrated)
    r = fill_sparse_channel(r)
    g = fill_sparse_channel(g)
    b = fill_sparse_channel(b)
    ir = fill_sparse_channel(ir)
    r, g, b, ir = apply_spectral_matrix(r, g, b, ir)
    mira_ndvi_native = compute_ndvi(ir, r)

    mira_rgb = np.stack([r, g, b], axis=-1)
    mira_rgb_u8 = (np.clip(mira_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    mira_png = cv2.imread(MIRA_PNG_PATH, cv2.IMREAD_UNCHANGED)
    if mira_png is None:
        raise FileNotFoundError(MIRA_PNG_PATH)

    mira_native_corners = order_corners(detect_marker_corners(to_gray(mira_rgb_u8)))
    mira_png_corners = order_corners(detect_marker_corners(to_gray(normalize_to_u8(mira_png))))

    homography, _ = cv2.findHomography(mira_native_corners, mira_png_corners)
    if homography is None:
        raise RuntimeError("Failed to compute MIRA raw-to-PNG homography")

    out_h, out_w = mira_png.shape[:2]
    mira_ndvi_png_frame = cv2.warpPerspective(
        mira_ndvi_native,
        homography,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
    )
    mira_mask = cv2.warpPerspective(
        np.full(mira_ndvi_native.shape, 255, dtype=np.uint8),
        homography,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
    ) > 0

    mapir_aligned = tiff.imread(MAPIR_ALIGNED_PATH).astype(np.float32)
    if mapir_aligned.ndim != 3 or mapir_aligned.shape[2] < 3:
        raise ValueError(f"{MAPIR_ALIGNED_PATH} must be a 3-channel image")

    mapir_r = mapir_aligned[:, :, 0]
    mapir_nir = mapir_aligned[:, :, 2]
    mapir_ndvi = compute_ndvi(mapir_nir, mapir_r)
    mapir_mask = cv2.imread(MAPIR_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if mapir_mask is None:
        raise FileNotFoundError(MAPIR_MASK_PATH)
    mapir_valid = mapir_mask > 0

    np.save(MIRA_NDVI_ARRAY_OUT, mira_ndvi_png_frame)
    np.save(MAPIR_NDVI_ARRAY_OUT, mapir_ndvi)

    save_heatmap(
        mira_ndvi_png_frame,
        MIRA_NDVI_HEATMAP_OUT,
        "MIRA NDVI (raw-derived, warped into PNG frame)",
        mask=mira_mask,
    )
    save_heatmap(
        mapir_ndvi,
        MAPIR_NDVI_HEATMAP_OUT,
        "MAPIR NDVI (aligned to MIRA PNG frame)",
        mask=mapir_valid,
    )

    print("Saved:", MIRA_NDVI_ARRAY_OUT)
    print("Saved:", MIRA_NDVI_HEATMAP_OUT)
    print("Saved:", MAPIR_NDVI_ARRAY_OUT)
    print("Saved:", MAPIR_NDVI_HEATMAP_OUT)
    print("MIRA output shape:", mira_ndvi_png_frame.shape)
    print("MAPIR output shape:", mapir_ndvi.shape)


if __name__ == "__main__":
    main()
