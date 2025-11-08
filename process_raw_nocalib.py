"""
Process RAW file from Mira220 RGB-IR camera
Pattern: [B G R G; G I G I; R G B G; G I G I] with horizontal flip
Optional spectral correction (enabled/disabled by argument)
No calibration (no black.raw or white.raw)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.colors import ListedColormap

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
    R_corr = R - 0.25*G - 0.5*I
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
        print("Usage: python process_raw.py <raw_filename> [--spectral-corr on|off]")
        sys.exit(1)

    raw_filename = sys.argv[1]
    spectral_flag = "on"
    if len(sys.argv) >= 4 and sys.argv[2] in ["--spectral-corr", "-s"]:
        spectral_flag = sys.argv[3].lower()

    print(f"[{get_timestamp()}] Starting Mira220 RGB-IR processing for {raw_filename}")
    raw_data = read_raw(raw_filename)
    print(f"[{get_timestamp()}] Skipping calibration (no black.raw or white.raw)")

    print(f"[{get_timestamp()}] Debayering with horizontal flip of pattern pixels")
    R, G, B, I = debayer_4x4_hflip(raw_data)

    if spectral_flag == "on":
        print(f"\n[{get_timestamp()}] Applying spectral cross-talk correction")
        R_corr, G_corr, B_corr = apply_spectral_correction(R, G, B, I)
    else:
        print(f"\n[{get_timestamp()}] Spectral correction disabled")
        R_corr, G_corr, B_corr = R, G, B

    rgb = build_rgb(R_corr, G_corr, B_corr)
    rgbir_mosaic = create_rgbir_mosaic(R_corr, G_corr, B_corr, I)
    ndvi = calculate_ndvi(I, R_corr)

    ndvi_mean, ndvi_median, ndvi_min, ndvi_max = ndvi.mean(), np.median(ndvi), ndvi.min(), ndvi.max()

    print(f"\n[{get_timestamp()}] NDVI Statistics:")
    print(f"  Mean: {ndvi_mean:.3f}, Median: {ndvi_median:.3f}")
    print(f"  Range: [{ndvi_min:.3f}, {ndvi_max:.3f}]")

    print(f"\n[{get_timestamp()}] Creating visualizations")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig = plt.figure(figsize=(16, 10))
    ax1 = plt.subplot(2, 3, 1); ax1.imshow(rgb); ax1.set_title('RGB Image'); ax1.axis('off')
    ax2 = plt.subplot(2, 3, 2); ax2.imshow(rgbir_mosaic); ax2.set_title('RGB-IR Mosaic'); ax2.axis('off')
    ax3 = plt.subplot(2, 3, 3); ax3.imshow(R_corr, cmap='Reds'); ax3.set_title('Red'); ax3.axis('off')
    ax4 = plt.subplot(2, 3, 4); ax4.imshow(G_corr, cmap='Greens'); ax4.set_title('Green'); ax4.axis('off')
    ax5 = plt.subplot(2, 3, 5); ax5.imshow(B_corr, cmap='Blues'); ax5.set_title('Blue'); ax5.axis('off')
    ax6 = plt.subplot(2, 3, 6); ax6.imshow(I, cmap='gray'); ax6.set_title('IR'); ax6.axis('off')
    plt.tight_layout()
    plt.savefig(f'{raw_filename}_Complete_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()

    colors_4 = ['red', 'yellow', 'lightgreen', 'darkgreen']
    cmap_4 = ListedColormap(colors_4)
    fig = plt.figure(figsize=(16, 5))
    ax1 = plt.subplot(1, 3, 1)
    im = ax1.imshow(ndvi, cmap=cmap_4, vmin=-1, vmax=1)
    ax1.set_title(f'NDVI False Color\nMean: {ndvi_mean:.3f}')
    ax1.axis('off')
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax2 = plt.subplot(1, 3, 2)
    ax2.hist(ndvi.flatten(), bins=100, color='green', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('NDVI'); ax2.set_ylabel('Frequency')
    ax2.axvline(ndvi_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {ndvi_mean:.3f}')
    ax2.axvline(ndvi_median, color='blue', linestyle='--', linewidth=2, label=f'Median: {ndvi_median:.3f}')
    ax2.legend()
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(rgb)
    veg_mask = ndvi > 0
    ndvi_overlay = np.ma.masked_where(~veg_mask, ndvi)
    im_overlay = ax3.imshow(ndvi_overlay, cmap=cmap_4, alpha=0.5, vmin=-1, vmax=1)
    ax3.set_title('RGB + NDVI Overlay (Vegetation)')
    ax3.axis('off')
    plt.colorbar(im_overlay, ax=ax3, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f'{raw_filename}_NDVI_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n[{get_timestamp()}] Processing complete for {raw_filename}")
    print(f"Spectral correction: {spectral_flag.upper()}")
