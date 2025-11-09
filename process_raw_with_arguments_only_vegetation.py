"""
Process RAW file from Mira220 RGB-IR camera
Pattern: [B G R G; G I G I; R G B G; G I G I] with horizontal flip
Optional spectral correction and calibration (black, white, both, none)
Vegetation masking and NDVI visualization with custom color thresholds.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.colors import ListedColormap, BoundaryNorm

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

def read_raw(filename, width=None, height=None):
    """Read raw file and auto-detect dimensions if not provided"""
    data = np.fromfile(filename, dtype=np.uint16)
    if width is None or height is None:
        total_pixels = len(data)
        possible_dims = [(1600, 1400), (1400, 1600), (1280, 1750), (1750, 1280)]
        for w, h in possible_dims:
            if w * h == total_pixels:
                width, height = w, h
                print(f"[{get_timestamp()}] Auto-detected dimensions: {width}×{height}")
                break
    return data.reshape((height, width))

def calibrate(raw_data, dark_frame=None, white_frame=None):
    """Apply dark/white calibration"""
    calibrated = raw_data.astype(np.float32)
    if dark_frame is not None:
        calibrated = calibrated - dark_frame.astype(np.float32)
    if white_frame is not None:
        white_mean = white_frame.mean()
        dark_mean = dark_frame.mean() if dark_frame is not None else 0
        calibrated = calibrated / (white_frame.astype(np.float32) - dark_mean) * (white_mean - dark_mean)
    return np.clip(calibrated, 0, None)

def debayer_4x4_hflip(raw_data):
    """Debayer 4x4 RGB-IR pattern with horizontal flip"""
    h, w = raw_data.shape
    out_h, out_w = h // 4, w // 4
    R = np.zeros((out_h, out_w), dtype=np.float32)
    G = np.zeros((out_h, out_w), dtype=np.float32)
    B = np.zeros((out_h, out_w), dtype=np.float32)
    I = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            block = raw_data[i*4:(i+1)*4, j*4:(j+1)*4].astype(np.float32)
            R[i, j] = (block[0, 1] + block[2, 3]) / 2.0
            B[i, j] = (block[0, 3] + block[2, 1]) / 2.0
            G[i, j] = (block[0, 0] + block[0, 2] + block[1, 1] + block[1, 3] +
                       block[2, 0] + block[2, 2] + block[3, 1] + block[3, 3]) / 8.0
            I[i, j] = (block[1, 0] + block[1, 2] + block[3, 0] + block[3, 2]) / 4.0
    return R, G, B, I

def apply_spectral_correction(R, G, B, I):
    """Apply spectral cross-talk correction"""
    R_corr = R - 0.4*G - 0.3*I
    G_corr = G - 0.3*B - 0.3*I
    B_corr = B - 0.15*R - 0.15*G - 0.4*I
    return np.clip(R_corr, 0, None), np.clip(G_corr, 0, None), np.clip(B_corr, 0, None)

def build_rgb(R, G, B, percentile=99.5):
    """Build RGB image with percentile normalization"""
    max_val = np.percentile(np.stack([R, G, B]), percentile)
    rgb = np.zeros((*R.shape, 3), dtype=np.float32)
    rgb[:, :, 0] = np.clip(R / max_val, 0, 1)
    rgb[:, :, 1] = np.clip(G / max_val, 0, 1)
    rgb[:, :, 2] = np.clip(B / max_val, 0, 1)
    return rgb

def create_rgbir_mosaic(R, G, B, I):
    """Create 2x2 mosaic of R, G, B, IR channels"""
    h, w = R.shape
    mosaic = np.zeros((h*2, w*2, 3), dtype=np.float32)
    max_rgb = np.percentile(np.stack([R, G, B]), 99.5)
    max_ir = np.percentile(I, 99.5)
    mosaic[:h, :w, 0] = np.clip(R / max_rgb, 0, 1)
    mosaic[:h, w:, 1] = np.clip(G / max_rgb, 0, 1)
    mosaic[h:, :w, 2] = np.clip(B / max_rgb, 0, 1)
    ir_norm = np.clip(I / max_ir, 0, 1)
    mosaic[h:, w:, :] = ir_norm[:, :, np.newaxis]
    return mosaic

def calculate_ndvi(I, R):
    """Calculate NDVI = (IR - R) / (IR + R)"""
    denominator = I + R
    ndvi = np.divide(I - R, denominator, where=denominator != 0)
    ndvi[denominator == 0] = 0
    return ndvi

# ============================================================================
# Main Processing
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_raw_with_arguments.py <raw_filename> [--calib none|black|white|both] [--spectral-corr on|off]")
        sys.exit(1)

    raw_filename = sys.argv[1]
    calib_flag = "none"
    spectral_flag = "on"

    if "--calib" in sys.argv:
        calib_flag = sys.argv[sys.argv.index("--calib") + 1].lower()
    if "--spectral-corr" in sys.argv:
        spectral_flag = sys.argv[sys.argv.index("--spectral-corr") + 1].lower()

    print(f"[{get_timestamp()}] Starting Mira220 RGB-IR processing for {raw_filename}")
    raw_data = read_raw(raw_filename)

    dark_frame = None
    white_frame = None
    if calib_flag in ["black", "both"]:
        print(f"[{get_timestamp()}] Loading black.raw for dark calibration")
        dark_frame = read_raw("black.raw")
    if calib_flag in ["white", "both"]:
        print(f"[{get_timestamp()}] Loading white.raw for white calibration")
        white_frame = read_raw("white.raw")

    calibrated_data = calibrate(raw_data, dark_frame, white_frame)
    print(f"[{get_timestamp()}] Debayering with horizontal flip")
    R, G, B, I = debayer_4x4_hflip(calibrated_data)

    if spectral_flag == "on":
        print(f"[{get_timestamp()}] Applying spectral correction")
        R_corr, G_corr, B_corr = apply_spectral_correction(R, G, B, I)
    else:
        print(f"[{get_timestamp()}] Spectral correction disabled")
        R_corr, G_corr, B_corr = R, G, B

    # Calculate NDVI
    ndvi = calculate_ndvi(I, R_corr)

    # Remove non-vegetation regions
    VEG_THRESHOLD = 0.2  # NDVI below this is treated as non-vegetation
    veg_mask = ndvi > VEG_THRESHOLD

    print(f"[{get_timestamp()}] Applying vegetation mask (NDVI > {VEG_THRESHOLD})")

    # Mask NDVI, RGB, and IR data
    ndvi_masked = np.where(veg_mask, ndvi, np.nan)
    R_masked = np.where(veg_mask, R_corr, 0)
    G_masked = np.where(veg_mask, G_corr, 0)
    B_masked = np.where(veg_mask, B_corr, 0)
    I_masked = np.where(veg_mask, I, 0)

    # Rebuild cleaned RGB image (vegetation only)
    rgb_masked = build_rgb(R_masked, G_masked, B_masked)

    # Recalculate statistics on vegetation only
    valid_ndvi = ndvi[veg_mask]
    ndvi_mean, ndvi_median, ndvi_min, ndvi_max = (
        np.nanmean(valid_ndvi),
        np.nanmedian(valid_ndvi),
        np.nanmin(valid_ndvi),
        np.nanmax(valid_ndvi),
    )

    print(f"\n[{get_timestamp()}] NDVI Statistics (Vegetation Only):")
    print(f"  Mean: {ndvi_mean:.3f}, Median: {ndvi_median:.3f}")
    print(f"  Range: [{ndvi_min:.3f}, {ndvi_max:.3f}]")

    # ======================================================================
    # NDVI Visualization (vegetation only, no numeric overlay)
    # ======================================================================
    print(f"[{get_timestamp()}] Creating NDVI visualization (vegetation only, no numeric overlay)")

    # Define NDVI color thresholds (below 0.1 = neutral gray)
    colors_ndvi = ['gray', 'red', 'yellow', 'lightgreen', 'darkgreen']
    bounds = [-1.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    cmap_ndvi = ListedColormap(colors_ndvi)
    norm_ndvi = BoundaryNorm(bounds, cmap_ndvi.N)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig = plt.figure(figsize=(16, 5))

    # NDVI False Color (vegetation only)
    ax1 = plt.subplot(1, 3, 1)
    im = ax1.imshow(ndvi_masked, cmap=cmap_ndvi, norm=norm_ndvi)
    ax1.set_title(f'NDVI False Color (Vegetation Only)\nMean: {ndvi_mean:.3f}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, ticks=bounds)

    # NDVI Histogram
    ax2 = plt.subplot(1, 3, 2)
    ax2.hist(valid_ndvi.flatten(), bins=100, color='green', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('NDVI')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'NDVI Distribution\nMedian: {ndvi_median:.3f}', fontsize=12, fontweight='bold')
    ax2.axvline(ndvi_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {ndvi_mean:.3f}')
    ax2.axvline(ndvi_median, color='blue', linestyle='--', linewidth=2, label=f'Median: {ndvi_median:.3f}')
    ax2.legend()

    # RGB + NDVI Overlay (vegetation only)
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(rgb_masked)
    ndvi_overlay = np.ma.masked_invalid(ndvi_masked)
    im_overlay = ax3.imshow(ndvi_overlay, cmap=cmap_ndvi, norm=norm_ndvi, alpha=0.5)
    ax3.set_title('RGB + NDVI Overlay (Vegetation Only)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im_overlay, ax=ax3, fraction=0.046, pad=0.04, ticks=bounds)

    plt.tight_layout()
    plt.savefig(f'{raw_filename}_NDVI_VegetationOnly_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[{get_timestamp()}] Processing complete for {raw_filename}")
    print(f"Spectral correction: {spectral_flag.upper()}")
    print(f"Calibration: {calib_flag.upper()}")
