"""
One-shot MIRA + MAPIR processing pipeline.

What this script does:
- reads a MIRA `.raw` image plus `dark.raw` and `white.raw`
- calibrates and debayers the MIRA data
- generates a MIRA RGB PNG from the raw input
- detects the ArUco marker in the generated MIRA PNG and the full-size MAPIR TIFF
- aligns and downsamples the MAPIR TIFF into the MIRA frame
- computes NDVI for both images
- writes NDVI arrays, NDVI heat maps, an alignment overlay, and a MIRA-MAPIR NDVI difference map

Typical usage:
    .venv/bin/python run_mira_mapir_pipeline.py \
      --mira-raw Vijay_0315_leaves_pic1.raw \
      --mapir-tif Vijay_0315_leaves_tif.tif \
      --dark-raw dark.raw \
      --white-raw white.raw \
      --output-prefix run1

Key outputs for `--output-prefix run1`:
- `run1_mira_rgb.png`
- `run1_mapir_aligned_to_mira.tif`
- `run1_mapir_alignment_overlay.png`
- `run1_mira_ndvi_heatmap.png`
- `run1_mapir_ndvi_heatmap.png`
- `run1_ndvi_difference_mira_minus_mapir_heatmap.png`

Notes:
- the output frame is the generated MIRA RGB image resolution
- this script expects ArUco marker ID 830 in both images
- MAPIR NDVI is computed as `(NIR - R) / (NIR + R)` using TIFF channels 3 and 1 in the aligned image
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import tifffile as tiff

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WIDTH = 1600
HEIGHT = 1400
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process one MIRA raw image and one full-size MAPIR TIFF in one run: "
            "build MIRA RGB/NDVI from raw, align the MAPIR TIFF to the MIRA PNG frame "
            "using the ArUco marker, and save both NDVI heatmaps."
        )
    )
    parser.add_argument("--mira-raw", required=True, help="Path to the MIRA .raw image")
    parser.add_argument("--mapir-tif", required=True, help="Path to the full-size MAPIR TIFF")
    parser.add_argument("--dark-raw", default="dark.raw", help="Dark calibration raw file")
    parser.add_argument("--white-raw", default="white.raw", help="White calibration raw file")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for output files. Defaults to the MIRA raw filename stem.",
    )
    return parser.parse_args()


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
    raw_f = raw.astype(np.float32)
    black_f = blockflip_4x4(black).astype(np.float32)
    white_f = blockflip_4x4(white).astype(np.float32)
    denom = white_f - black_f
    denom[denom == 0] = 1.0
    return np.clip((raw_f - black_f) / denom, 0.0, 1.0)


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
    original = result.copy()
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

        result[0, :] = original[0, :]
        result[-1, :] = original[-1, :]
        result[:, 0] = original[:, 0]
        result[:, -1] = original[:, -1]
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


def save_rgb_png(image: np.ndarray, out_path: Path) -> None:
    rgb_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)


def save_heatmap(ndvi: np.ndarray, out_path: Path, title: str, mask: np.ndarray | None = None) -> None:
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


def make_overlay(mira_png: np.ndarray, aligned_mapir_u8: np.ndarray) -> np.ndarray:
    mira_u8 = normalize_to_u8(mira_png)
    if mira_u8.ndim == 2:
        mira_u8 = cv2.cvtColor(mira_u8, cv2.COLOR_GRAY2BGR)
    elif mira_u8.shape[2] == 4:
        mira_u8 = cv2.cvtColor(mira_u8, cv2.COLOR_BGRA2BGR)

    if aligned_mapir_u8.ndim == 2:
        aligned_mapir_u8 = cv2.cvtColor(aligned_mapir_u8, cv2.COLOR_GRAY2BGR)

    overlay = np.zeros_like(mira_u8[:, :, :3])
    overlay[:, :, 1] = mira_u8[:, :, 1]
    overlay[:, :, 0] = aligned_mapir_u8[:, :, 0]
    overlay[:, :, 2] = aligned_mapir_u8[:, :, 2]
    return overlay


def main() -> None:
    args = parse_args()
    prefix = Path(args.output_prefix) if args.output_prefix else Path(args.mira_raw).with_suffix("")

    out_mira_png = Path(f"{prefix}_mira_rgb.png")
    out_mapir_aligned_tif = Path(f"{prefix}_mapir_aligned_to_mira.tif")
    out_mapir_mask = Path(f"{prefix}_mapir_aligned_mask.png")
    out_overlay = Path(f"{prefix}_mapir_alignment_overlay.png")
    out_mira_ndvi = Path(f"{prefix}_mira_ndvi.npy")
    out_mapir_ndvi = Path(f"{prefix}_mapir_ndvi.npy")
    out_mira_heatmap = Path(f"{prefix}_mira_ndvi_heatmap.png")
    out_mapir_heatmap = Path(f"{prefix}_mapir_ndvi_heatmap.png")
    out_diff_ndvi = Path(f"{prefix}_ndvi_difference_mira_minus_mapir.npy")
    out_diff_heatmap = Path(f"{prefix}_ndvi_difference_mira_minus_mapir_heatmap.png")
    out_overlap_mask = Path(f"{prefix}_ndvi_overlap_mask.png")

    mira_raw = read_raw(args.mira_raw)
    dark_raw = read_raw(args.dark_raw)
    white_raw = read_raw(args.white_raw)

    calibrated = calibrate_bayer(mira_raw, dark_raw, white_raw)
    r, g, b, ir = debayer_mira(calibrated)
    r = fill_sparse_channel(r)
    g = fill_sparse_channel(g)
    b = fill_sparse_channel(b)
    ir = fill_sparse_channel(ir)
    r, g, b, ir = apply_spectral_matrix(r, g, b, ir)

    mira_rgb = np.stack([r, g, b], axis=-1)
    mira_ndvi_native = compute_ndvi(ir, r)
    save_rgb_png(mira_rgb, out_mira_png)

    mira_png = cv2.imread(str(out_mira_png), cv2.IMREAD_UNCHANGED)
    mapir_image = tiff.imread(args.mapir_tif)
    if mira_png is None:
        raise FileNotFoundError(out_mira_png)
    if mapir_image is None:
        raise FileNotFoundError(args.mapir_tif)

    mira_gray = to_gray(normalize_to_u8(mira_png))
    mapir_gray = to_gray(normalize_to_u8(mapir_image))
    mira_corners = order_corners(detect_marker_corners(mira_gray))
    mapir_corners = order_corners(detect_marker_corners(mapir_gray))

    mapir_to_mira_h, _ = cv2.findHomography(mapir_corners, mira_corners)
    if mapir_to_mira_h is None:
        raise RuntimeError("Failed to compute MAPIR-to-MIRA homography")

    out_h, out_w = mira_gray.shape
    aligned_mapir = cv2.warpPerspective(
        mapir_image,
        mapir_to_mira_h,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
    )
    coverage_mask = cv2.warpPerspective(
        np.full(mapir_gray.shape, 255, dtype=np.uint8),
        mapir_to_mira_h,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
    )

    tiff.imwrite(str(out_mapir_aligned_tif), aligned_mapir)
    cv2.imwrite(str(out_mapir_mask), coverage_mask)
    cv2.imwrite(str(out_overlay), make_overlay(mira_png, normalize_to_u8(aligned_mapir)))

    mira_native_corners = order_corners(
        detect_marker_corners(to_gray((np.clip(mira_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)))
    )
    png_corners = order_corners(detect_marker_corners(mira_gray))
    native_to_png_h, _ = cv2.findHomography(mira_native_corners, png_corners)
    if native_to_png_h is None:
        raise RuntimeError("Failed to compute MIRA native-to-PNG homography")

    mira_ndvi_png_frame = cv2.warpPerspective(
        mira_ndvi_native,
        native_to_png_h,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
    )
    mira_mask = cv2.warpPerspective(
        np.full(mira_ndvi_native.shape, 255, dtype=np.uint8),
        native_to_png_h,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
    ) > 0

    aligned_mapir = aligned_mapir.astype(np.float32)
    if aligned_mapir.ndim != 3 or aligned_mapir.shape[2] < 3:
        raise ValueError("Aligned MAPIR image must be a 3-channel image")
    mapir_r = aligned_mapir[:, :, 0]
    mapir_nir = aligned_mapir[:, :, 2]
    mapir_ndvi = compute_ndvi(mapir_nir, mapir_r)
    mapir_valid = coverage_mask > 0

    np.save(out_mira_ndvi, mira_ndvi_png_frame)
    np.save(out_mapir_ndvi, mapir_ndvi)
    save_heatmap(out_path=out_mira_heatmap, ndvi=mira_ndvi_png_frame, title="MIRA NDVI", mask=mira_mask)
    save_heatmap(out_path=out_mapir_heatmap, ndvi=mapir_ndvi, title="MAPIR NDVI aligned to MIRA", mask=mapir_valid)

    overlap_mask = mira_mask & mapir_valid & np.isfinite(mira_ndvi_png_frame) & np.isfinite(mapir_ndvi)
    diff_ndvi = np.full_like(mira_ndvi_png_frame, np.nan, dtype=np.float32)
    diff_ndvi[overlap_mask] = mira_ndvi_png_frame[overlap_mask] - mapir_ndvi[overlap_mask]
    np.save(out_diff_ndvi, diff_ndvi)
    cv2.imwrite(str(out_overlap_mask), overlap_mask.astype(np.uint8) * 255)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(diff_ndvi, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("NDVI Difference: MIRA - MAPIR")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NDVI delta")
    fig.tight_layout()
    fig.savefig(out_diff_heatmap, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_mira_png)
    print("Saved:", out_mapir_aligned_tif)
    print("Saved:", out_mapir_mask)
    print("Saved:", out_overlay)
    print("Saved:", out_mira_ndvi)
    print("Saved:", out_mapir_ndvi)
    print("Saved:", out_mira_heatmap)
    print("Saved:", out_mapir_heatmap)
    print("Saved:", out_diff_ndvi)
    print("Saved:", out_diff_heatmap)
    print("Saved:", out_overlap_mask)
    print("Output resolution:", (out_w, out_h))


if __name__ == "__main__":
    main()
