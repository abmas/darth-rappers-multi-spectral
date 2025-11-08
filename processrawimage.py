"""

Process Leaves.raw with Mira220 RGB-IR camera

Pattern: [B G R G; G I G I; R G B G; G I G I] with horizontal flip

Spectral correction:

R_corrected = R - 0.25×G - 0.5×IR

G_corrected = G - 0.3×B - 0.3×IR  

B_corrected = B - 0.15×R - 0.15×G - 0.4×IR

"""

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime

import os

 

def get_timestamp():

    """Return formatted timestamp string"""

    return datetime.now().strftime('%H:%M:%S')

 

def read_raw(path, w=None, h=None):

    """Read raw uint16 file with auto-detection"""

    file_size = os.path.getsize(path)

    total_pixels = file_size // 2

   

    if w is None or h is None:

        common_dims = [(2592, 1944), (2464, 2048), (1600, 1400), (2048, 2464), (1944, 2592)]

        for width, height in common_dims:

            if width * height == total_pixels:

                w, h = width, height

                print(f"[{get_timestamp()}] Auto-detected dimensions: {w}×{h}")

                break

        if w is None:

            raise ValueError(f"Cannot auto-detect dimensions. File has {total_pixels} pixels.")

   

    raw = np.fromfile(path, dtype=np.uint16)

    return raw.reshape((h, w))

 

def calibrate(raw, dark, white):

    """Apply dark frame subtraction and white reference normalization"""

    if raw.shape != dark.shape or raw.shape != white.shape:

        raise ValueError(f"Shape mismatch: raw{raw.shape}, dark{dark.shape}, white{white.shape}")

   

    white_minus_dark = white.astype(np.float32) - dark.astype(np.float32)

    white_minus_dark[white_minus_dark <= 0] = 1

   

    calibrated = (raw.astype(np.float32) - dark.astype(np.float32)) / white_minus_dark

    calibrated *= np.mean(white_minus_dark)

    calibrated = np.clip(calibrated, 0, None)

   

    return calibrated

 

def debayer_hflip(raw):

    """

    Debayer with horizontal flip of pattern:

    Original: [B G R G; G I G I; R G B G; G I G I]

    Flipped:  [G R G B; I G I G; G B G R; I G I G]

    """

    h, w = raw.shape

    h_out, w_out = h // 4, w // 4

   

    R = np.zeros((h_out, w_out), dtype=np.float32)

    G = np.zeros((h_out, w_out), dtype=np.float32)

    B = np.zeros((h_out, w_out), dtype=np.float32)

    I = np.zeros((h_out, w_out), dtype=np.float32)

   

    for i in range(h_out):

        for j in range(w_out):

            block = raw[i*4:(i+1)*4, j*4:(j+1)*4]

            # After horizontal flip:

            # Row 0: G R G B

            # Row 1: I G I G

            # Row 2: G B G R

            # Row 3: I G I G

            R[i, j] = (block[0, 1] + block[2, 3]) / 2.0

            B[i, j] = (block[0, 3] + block[2, 1]) / 2.0

            G[i, j] = (block[0, 0] + block[0, 2] + block[1, 1] + block[1, 3] +

                       block[2, 0] + block[2, 2] + block[3, 1] + block[3, 3]) / 8.0

            I[i, j] = (block[1, 0] + block[1, 2] + block[3, 0] + block[3, 2]) / 4.0

   

    return R, G, B, I

 

def apply_spectral_correction(R, G, B, I):

    """

    Apply spectral cross-talk correction

    R_corrected = R - 0.25×G - 0.5×IR

    G_corrected = G - 0.3×B - 0.3×IR  

    B_corrected = B - 0.15×R - 0.15×G - 0.4×IR

    """

    R_corr = R - 0.25*G - 0.5*I

    G_corr = G - 0.3*B - 0.3*I

    B_corr = B - 0.15*R - 0.15*G - 0.4*I

   

    R_corr = np.clip(R_corr, 0, None)

    G_corr = np.clip(G_corr, 0, None)

    B_corr = np.clip(B_corr, 0, None)

   

    return R_corr, G_corr, B_corr

 

def build_rgb(R, G, B):

    """Build RGB image with normalization"""

    rgb = np.dstack([R, G, B])

    p99 = np.percentile(rgb, 99.5)

    if p99 > 0:

        rgb = rgb / p99

    rgb = np.clip(rgb, 0, 1)

    return rgb

 

def create_rgbir_mosaic(R, G, B, I):

    """Create RGB-IR mosaic visualization (2x2 grid)"""

    def norm(channel):

        p99 = np.percentile(channel, 99.5)

        return np.clip(channel / p99 if p99 > 0 else channel, 0, 1)

   

    R_norm = norm(R)

    G_norm = norm(G)

    B_norm = norm(B)

    I_norm = norm(I)

   

    h, w = R.shape

    mosaic = np.zeros((h*2, w*2, 3))

   

    # Top-left: Red

    mosaic[0:h, 0:w, 0] = R_norm

    # Top-right: Green

    mosaic[0:h, w:2*w, 1] = G_norm

    # Bottom-left: Blue

    mosaic[h:2*h, 0:w, 2] = B_norm

    # Bottom-right: IR (grayscale)

    mosaic[h:2*h, w:2*w, 0] = I_norm

    mosaic[h:2*h, w:2*w, 1] = I_norm

    mosaic[h:2*h, w:2*w, 2] = I_norm

   

    return mosaic

 

# ============================================================================

# Main Processing

# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

 

print(f"[{get_timestamp()}] Starting Mira220 RGB-IR processing")

print(f"[{get_timestamp()}] Loading Leaves.raw")

raw = read_raw(os.path.join(script_dir, 'Leaves.raw'))

 

print(f"[{get_timestamp()}] Loading calibration frames")

dark = read_raw(os.path.join(script_dir, 'black.raw'), raw.shape[1], raw.shape[0])

white = read_raw(os.path.join(script_dir, 'white.raw'), raw.shape[1], raw.shape[0])

 

print(f"[{get_timestamp()}] Applying dark/white calibration")

raw_cal = calibrate(raw, dark, white)

 

print(f"[{get_timestamp()}] Debayering with horizontal flip pattern")

R, G, B, I = debayer_hflip(raw_cal)

 

print(f"\n[{get_timestamp()}] Raw channel means:")

print(f"  R: {R.mean():.1f}, G: {G.mean():.1f}, B: {B.mean():.1f}, IR: {I.mean():.1f}")

 

print(f"\n[{get_timestamp()}] Applying spectral cross-talk correction:")

print(f"  R_corrected = R - 0.25×G - 0.5×IR")

print(f"  G_corrected = G - 0.3×B - 0.3×IR")

print(f"  B_corrected = B - 0.15×R - 0.15×G - 0.4×IR")

 

R_corr, G_corr, B_corr = apply_spectral_correction(R, G, B, I)

 

print(f"\n[{get_timestamp()}] Corrected channel means:")

print(f"  R: {R_corr.mean():.1f}, G: {G_corr.mean():.1f}, B: {B_corr.mean():.1f}")

print(f"  Color ratios: R/G={R_corr.mean()/G_corr.mean():.3f}, B/G={B_corr.mean()/G_corr.mean():.3f}")

 

# Build RGB image

print(f"\n[{get_timestamp()}] Building RGB image")

rgb = build_rgb(R_corr, G_corr, B_corr)

 

# Create RGB-IR mosaic

print(f"[{get_timestamp()}] Creating RGB-IR mosaic")

rgbir_mosaic = create_rgbir_mosaic(R_corr, G_corr, B_corr, I)

 

# Calculate NDVI

print(f"[{get_timestamp()}] Calculating NDVI")

epsilon = 1e-10

ndvi = (I - R_corr) / (I + R_corr + epsilon)

ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)

 

ndvi_mean = ndvi.mean()

ndvi_median = np.median(ndvi)

ndvi_min = ndvi.min()

ndvi_max = ndvi.max()

 

print(f"\n[{get_timestamp()}] NDVI Statistics:")

print(f"  Mean: {ndvi_mean:.3f}, Median: {ndvi_median:.3f}")

print(f"  Range: [{ndvi_min:.3f}, {ndvi_max:.3f}]")

 

# ============================================================================

# Visualization

# ============================================================================

print(f"\n[{get_timestamp()}] Creating visualizations")

 

fig = plt.figure(figsize=(16, 10))

 

# 1. RGB Image

ax1 = plt.subplot(2, 3, 1)

ax1.imshow(rgb)

ax1.set_title('RGB Image (Spectral Corrected)\n' +

              'R-0.25G-0.5IR, G-0.3B-0.3IR, B-0.15R-0.15G-0.4IR',

              fontsize=11, fontweight='bold')

ax1.axis('off')

 

# 2. RGB-IR Mosaic

ax2 = plt.subplot(2, 3, 2)

ax2.imshow(rgbir_mosaic)

ax2.set_title('RGB-IR Mosaic\nR | G\n---+---\nB | IR',

              fontsize=12, fontweight='bold')

ax2.axis('off')

 

# 3. Red channel

ax3 = plt.subplot(2, 3, 3)

ax3.imshow(R_corr, cmap='Reds')

ax3.set_title(f'Red Channel (corrected)\nMean: {R_corr.mean():.1f}',

              fontsize=11, fontweight='bold')

ax3.axis('off')

 

# 4. Green channel

ax4 = plt.subplot(2, 3, 4)

ax4.imshow(G_corr, cmap='Greens')

ax4.set_title(f'Green Channel (corrected)\nMean: {G_corr.mean():.1f}',

              fontsize=11, fontweight='bold')

ax4.axis('off')

 

# 5. Blue channel

ax5 = plt.subplot(2, 3, 5)

ax5.imshow(B_corr, cmap='Blues')

ax5.set_title(f'Blue Channel (corrected)\nMean: {B_corr.mean():.1f}',

              fontsize=11, fontweight='bold')

ax5.axis('off')

 

# 6. IR channel

ax6 = plt.subplot(2, 3, 6)

ax6.imshow(I, cmap='gray')

ax6.set_title(f'IR Channel (raw)\nMean: {I.mean():.1f}',

              fontsize=11, fontweight='bold')

ax6.axis('off')

 

plt.suptitle(f'Mira220 RGB-IR Processing - Leaves.raw\n' +

             f'Pattern: [B G R G; G I G I; R G B G; G I G I] (H-Flipped)\n' +

             f'Spectral Correction Applied | NDVI Mean: {ndvi_mean:.3f} | Timestamp: {timestamp}',

             fontsize=13, fontweight='bold')

plt.tight_layout()

 

out_path = os.path.join(script_dir, f'Leaves_RGBIR_Complete_{timestamp}.png')

plt.savefig(out_path, dpi=150, bbox_inches='tight')

plt.close()

 

print(f"[{get_timestamp()}] Saved: {out_path}")

 

# Save individual RGB image

fig_rgb = plt.figure(figsize=(12, 10))

plt.imshow(rgb)

plt.title(f'Leaves.raw - RGB Demosaiced (Spectral Corrected)\n' +

          f'R_corrected = R - 0.25×G - 0.5×IR\n' +

          f'G_corrected = G - 0.3×B - 0.3×IR\n' +

          f'B_corrected = B - 0.15×R - 0.15×G - 0.4×IR\n' +

          f'Timestamp: {timestamp}',

          fontsize=14, fontweight='bold')

plt.axis('off')

 

out_rgb = os.path.join(script_dir, f'Leaves_RGB_{timestamp}.png')

plt.savefig(out_rgb, dpi=150, bbox_inches='tight')

plt.close()

 

print(f"[{get_timestamp()}] Saved: {out_rgb}")

 

# Save RGB-IR mosaic

fig_mosaic = plt.figure(figsize=(12, 10))

plt.imshow(rgbir_mosaic)

plt.title(f'Leaves.raw - RGB-IR Mosaic\n' +

          f'R (top-left) | G (top-right)\n' +

          f'B (bottom-left) | IR (bottom-right)\n' +

          f'Timestamp: {timestamp}',

          fontsize=14, fontweight='bold')

plt.axis('off')

 

out_mosaic = os.path.join(script_dir, f'Leaves_RGBIR_Mosaic_{timestamp}.png')

plt.savefig(out_mosaic, dpi=150, bbox_inches='tight')

plt.close()

 

print(f"[{get_timestamp()}] Saved: {out_mosaic}")

 

# Save NDVI visualization

print(f"[{get_timestamp()}] Creating NDVI visualization")

fig_ndvi = plt.figure(figsize=(14, 6))

 

# NDVI false color

ax1 = plt.subplot(1, 2, 1)

im = ax1.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)

ax1.set_title(f'NDVI False Color\nMean: {ndvi_mean:.3f}, Median: {ndvi_median:.3f}',

              fontsize=12, fontweight='bold')

ax1.axis('off')

plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='NDVI')

 

# NDVI histogram

ax2 = plt.subplot(1, 2, 2)

ax2.hist(ndvi.flatten(), bins=100, color='green', alpha=0.7, edgecolor='black')

ax2.axvline(ndvi_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {ndvi_mean:.3f}')

ax2.axvline(ndvi_median, color='blue', linestyle='--', linewidth=2, label=f'Median: {ndvi_median:.3f}')

ax2.set_xlabel('NDVI Value', fontsize=11, fontweight='bold')

ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')

ax2.set_title('NDVI Distribution', fontsize=12, fontweight='bold')

ax2.legend()

ax2.grid(True, alpha=0.3)

 

plt.suptitle(f'NDVI Analysis - Leaves.raw\nRange: [{ndvi_min:.3f}, {ndvi_max:.3f}] | Timestamp: {timestamp}',

             fontsize=13, fontweight='bold')

plt.tight_layout()

 

out_ndvi = os.path.join(script_dir, f'Leaves_NDVI_{timestamp}.png')

plt.savefig(out_ndvi, dpi=150, bbox_inches='tight')

plt.close()

 

print(f"[{get_timestamp()}] Saved: {out_ndvi}")

 

print(f"\n[{get_timestamp()}] Processing complete!")

print("\nSPECTRAL CORRECTION SUMMARY:")

print("="*80)

print("Applied cross-talk correction coefficients:")

print("  R_corrected = R - 0.25×G - 0.5×IR  (removes green + IR contamination)")

print("  G_corrected = G - 0.3×B - 0.3×IR   (removes blue + IR contamination)")

print("  B_corrected = B - 0.15×R - 0.15×G - 0.4×IR  (removes red + green + IR)")

print("="*80)

print("\nNDVI RESULTS:")

print(f"  Formula: NDVI = (IR - R_corrected) / (IR + R_corrected)")

print(f"  Mean: {ndvi_mean:.3f}")

print(f"  Median: {ndvi_median:.3f}")

print(f"  Range: [{ndvi_min:.3f}, {ndvi_max:.3f}]")

print("="*80)

print("\nNote: Spectral response curves from Mira220_DS000642_9-00.pdf datasheet")

print("were used as reference for cross-talk correction coefficients.")

print("="*80)﻿
"""

Process Leaves.raw with Mira220 RGB-IR camera

Pattern: [B G R G; G I G I; R G B G; G I G I] with horizontal flip

Spectral correction:

R_corrected = R - 0.25×G - 0.5×IR

G_corrected = G - 0.3×B - 0.3×IR  

B_corrected = B - 0.15×R - 0.15×G - 0.4×IR

"""

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime

import os

 

def get_timestamp():

    """Return formatted timestamp string"""

    return datetime.now().strftime('%H:%M:%S')

 

def read_raw(path, w=None, h=None):

    """Read raw uint16 file with auto-detection"""

    file_size = os.path.getsize(path)

    total_pixels = file_size // 2

   

    if w is None or h is None:

        common_dims = [(2592, 1944), (2464, 2048), (1600, 1400), (2048, 2464), (1944, 2592)]

        for width, height in common_dims:

            if width * height == total_pixels:

                w, h = width, height

                print(f"[{get_timestamp()}] Auto-detected dimensions: {w}×{h}")

                break

        if w is None:

            raise ValueError(f"Cannot auto-detect dimensions. File has {total_pixels} pixels.")

   

    raw = np.fromfile(path, dtype=np.uint16)

    return raw.reshape((h, w))

 

def calibrate(raw, dark, white):

    """Apply dark frame subtraction and white reference normalization"""

    if raw.shape != dark.shape or raw.shape != white.shape:

        raise ValueError(f"Shape mismatch: raw{raw.shape}, dark{dark.shape}, white{white.shape}")

   

    white_minus_dark = white.astype(np.float32) - dark.astype(np.float32)

    white_minus_dark[white_minus_dark <= 0] = 1

   

    calibrated = (raw.astype(np.float32) - dark.astype(np.float32)) / white_minus_dark

    calibrated *= np.mean(white_minus_dark)

    calibrated = np.clip(calibrated, 0, None)

   

    return calibrated

 

def debayer_hflip(raw):

    """

    Debayer with horizontal flip of pattern:

    Original: [B G R G; G I G I; R G B G; G I G I]

    Flipped:  [G R G B; I G I G; G B G R; I G I G]

    """

    h, w = raw.shape

    h_out, w_out = h // 4, w // 4

   

    R = np.zeros((h_out, w_out), dtype=np.float32)

    G = np.zeros((h_out, w_out), dtype=np.float32)

    B = np.zeros((h_out, w_out), dtype=np.float32)

    I = np.zeros((h_out, w_out), dtype=np.float32)

   

    for i in range(h_out):

        for j in range(w_out):

            block = raw[i*4:(i+1)*4, j*4:(j+1)*4]

            # After horizontal flip:

            # Row 0: G R G B

            # Row 1: I G I G

            # Row 2: G B G R

            # Row 3: I G I G

            R[i, j] = (block[0, 1] + block[2, 3]) / 2.0

            B[i, j] = (block[0, 3] + block[2, 1]) / 2.0

            G[i, j] = (block[0, 0] + block[0, 2] + block[1, 1] + block[1, 3] +

                       block[2, 0] + block[2, 2] + block[3, 1] + block[3, 3]) / 8.0

            I[i, j] = (block[1, 0] + block[1, 2] + block[3, 0] + block[3, 2]) / 4.0

   

    return R, G, B, I

 

def apply_spectral_correction(R, G, B, I):

    """

    Apply spectral cross-talk correction

    R_corrected = R - 0.25×G - 0.5×IR

    G_corrected = G - 0.3×B - 0.3×IR  

    B_corrected = B - 0.15×R - 0.15×G - 0.4×IR

    """

    R_corr = R - 0.25*G - 0.5*I

    G_corr = G - 0.3*B - 0.3*I

    B_corr = B - 0.15*R - 0.15*G - 0.4*I

   

    R_corr = np.clip(R_corr, 0, None)

    G_corr = np.clip(G_corr, 0, None)

    B_corr = np.clip(B_corr, 0, None)

   

    return R_corr, G_corr, B_corr

 

def build_rgb(R, G, B):

    """Build RGB image with normalization"""

    rgb = np.dstack([R, G, B])

    p99 = np.percentile(rgb, 99.5)

    if p99 > 0:

        rgb = rgb / p99

    rgb = np.clip(rgb, 0, 1)

    return rgb

 

def create_rgbir_mosaic(R, G, B, I):

    """Create RGB-IR mosaic visualization (2x2 grid)"""

    def norm(channel):

        p99 = np.percentile(channel, 99.5)

        return np.clip(channel / p99 if p99 > 0 else channel, 0, 1)

   

    R_norm = norm(R)

    G_norm = norm(G)

    B_norm = norm(B)

    I_norm = norm(I)

   

    h, w = R.shape

    mosaic = np.zeros((h*2, w*2, 3))

   

    # Top-left: Red

    mosaic[0:h, 0:w, 0] = R_norm

    # Top-right: Green

    mosaic[0:h, w:2*w, 1] = G_norm

    # Bottom-left: Blue

    mosaic[h:2*h, 0:w, 2] = B_norm

    # Bottom-right: IR (grayscale)

    mosaic[h:2*h, w:2*w, 0] = I_norm

    mosaic[h:2*h, w:2*w, 1] = I_norm

    mosaic[h:2*h, w:2*w, 2] = I_norm

   

    return mosaic

 

# ============================================================================

# Main Processing

# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

 

print(f"[{get_timestamp()}] Starting Mira220 RGB-IR processing")

print(f"[{get_timestamp()}] Loading Leaves.raw")

raw = read_raw(os.path.join(script_dir, 'Leaves.raw'))

 

print(f"[{get_timestamp()}] Loading calibration frames")

dark = read_raw(os.path.join(script_dir, 'black.raw'), raw.shape[1], raw.shape[0])

white = read_raw(os.path.join(script_dir, 'white.raw'), raw.shape[1], raw.shape[0])

 

print(f"[{get_timestamp()}] Applying dark/white calibration")

raw_cal = calibrate(raw, dark, white)

 

print(f"[{get_timestamp()}] Debayering with horizontal flip pattern")

R, G, B, I = debayer_hflip(raw_cal)

 

print(f"\n[{get_timestamp()}] Raw channel means:")

print(f"  R: {R.mean():.1f}, G: {G.mean():.1f}, B: {B.mean():.1f}, IR: {I.mean():.1f}")

 

print(f"\n[{get_timestamp()}] Applying spectral cross-talk correction:")

print(f"  R_corrected = R - 0.25×G - 0.5×IR")

print(f"  G_corrected = G - 0.3×B - 0.3×IR")

print(f"  B_corrected = B - 0.15×R - 0.15×G - 0.4×IR")

 

R_corr, G_corr, B_corr = apply_spectral_correction(R, G, B, I)

 

print(f"\n[{get_timestamp()}] Corrected channel means:")

print(f"  R: {R_corr.mean():.1f}, G: {G_corr.mean():.1f}, B: {B_corr.mean():.1f}")

print(f"  Color ratios: R/G={R_corr.mean()/G_corr.mean():.3f}, B/G={B_corr.mean()/G_corr.mean():.3f}")

 

# Build RGB image

print(f"\n[{get_timestamp()}] Building RGB image")

rgb = build_rgb(R_corr, G_corr, B_corr)

 

# Create RGB-IR mosaic

print(f"[{get_timestamp()}] Creating RGB-IR mosaic")

rgbir_mosaic = create_rgbir_mosaic(R_corr, G_corr, B_corr, I)

 

# Calculate NDVI

print(f"[{get_timestamp()}] Calculating NDVI")

epsilon = 1e-10

ndvi = (I - R_corr) / (I + R_corr + epsilon)

ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)

 

ndvi_mean = ndvi.mean()

ndvi_median = np.median(ndvi)

ndvi_min = ndvi.min()

ndvi_max = ndvi.max()

 

print(f"\n[{get_timestamp()}] NDVI Statistics:")

print(f"  Mean: {ndvi_mean:.3f}, Median: {ndvi_median:.3f}")

print(f"  Range: [{ndvi_min:.3f}, {ndvi_max:.3f}]")

 

# ============================================================================

# Visualization

# ============================================================================

print(f"\n[{get_timestamp()}] Creating visualizations")

 

fig = plt.figure(figsize=(16, 10))

 

# 1. RGB Image

ax1 = plt.subplot(2, 3, 1)

ax1.imshow(rgb)

ax1.set_title('RGB Image (Spectral Corrected)\n' +

              'R-0.25G-0.5IR, G-0.3B-0.3IR, B-0.15R-0.15G-0.4IR',

              fontsize=11, fontweight='bold')

ax1.axis('off')

 

# 2. RGB-IR Mosaic

ax2 = plt.subplot(2, 3, 2)

ax2.imshow(rgbir_mosaic)

ax2.set_title('RGB-IR Mosaic\nR | G\n---+---\nB | IR',

              fontsize=12, fontweight='bold')

ax2.axis('off')

 

# 3. Red channel

ax3 = plt.subplot(2, 3, 3)

ax3.imshow(R_corr, cmap='Reds')

ax3.set_title(f'Red Channel (corrected)\nMean: {R_corr.mean():.1f}',

              fontsize=11, fontweight='bold')

ax3.axis('off')

 

# 4. Green channel

ax4 = plt.subplot(2, 3, 4)

ax4.imshow(G_corr, cmap='Greens')

ax4.set_title(f'Green Channel (corrected)\nMean: {G_corr.mean():.1f}',

              fontsize=11, fontweight='bold')

ax4.axis('off')

 

# 5. Blue channel

ax5 = plt.subplot(2, 3, 5)

ax5.imshow(B_corr, cmap='Blues')

ax5.set_title(f'Blue Channel (corrected)\nMean: {B_corr.mean():.1f}',

              fontsize=11, fontweight='bold')

ax5.axis('off')

 

# 6. IR channel

ax6 = plt.subplot(2, 3, 6)

ax6.imshow(I, cmap='gray')

ax6.set_title(f'IR Channel (raw)\nMean: {I.mean():.1f}',

              fontsize=11, fontweight='bold')

ax6.axis('off')

 

plt.suptitle(f'Mira220 RGB-IR Processing - Leaves.raw\n' +

             f'Pattern: [B G R G; G I G I; R G B G; G I G I] (H-Flipped)\n' +

             f'Spectral Correction Applied | NDVI Mean: {ndvi_mean:.3f} | Timestamp: {timestamp}',

             fontsize=13, fontweight='bold')

plt.tight_layout()

 

out_path = os.path.join(script_dir, f'Leaves_RGBIR_Complete_{timestamp}.png')

plt.savefig(out_path, dpi=150, bbox_inches='tight')

plt.close()

 

print(f"[{get_timestamp()}] Saved: {out_path}")

 

# Save individual RGB image

fig_rgb = plt.figure(figsize=(12, 10))

plt.imshow(rgb)

plt.title(f'Leaves.raw - RGB Demosaiced (Spectral Corrected)\n' +

          f'R_corrected = R - 0.25×G - 0.5×IR\n' +

          f'G_corrected = G - 0.3×B - 0.3×IR\n' +

          f'B_corrected = B - 0.15×R - 0.15×G - 0.4×IR\n' +

          f'Timestamp: {timestamp}',

          fontsize=14, fontweight='bold')

plt.axis('off')

 

out_rgb = os.path.join(script_dir, f'Leaves_RGB_{timestamp}.png')

plt.savefig(out_rgb, dpi=150, bbox_inches='tight')

plt.close()

 

print(f"[{get_timestamp()}] Saved: {out_rgb}")

 

# Save RGB-IR mosaic

fig_mosaic = plt.figure(figsize=(12, 10))

plt.imshow(rgbir_mosaic)

plt.title(f'Leaves.raw - RGB-IR Mosaic\n' +

          f'R (top-left) | G (top-right)\n' +

          f'B (bottom-left) | IR (bottom-right)\n' +

          f'Timestamp: {timestamp}',

          fontsize=14, fontweight='bold')

plt.axis('off')

 

out_mosaic = os.path.join(script_dir, f'Leaves_RGBIR_Mosaic_{timestamp}.png')

plt.savefig(out_mosaic, dpi=150, bbox_inches='tight')

plt.close()

 

print(f"[{get_timestamp()}] Saved: {out_mosaic}")

 

# Save NDVI visualization

print(f"[{get_timestamp()}] Creating NDVI visualization")

fig_ndvi = plt.figure(figsize=(14, 6))

 

# NDVI false color

ax1 = plt.subplot(1, 2, 1)

im = ax1.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)

ax1.set_title(f'NDVI False Color\nMean: {ndvi_mean:.3f}, Median: {ndvi_median:.3f}',

              fontsize=12, fontweight='bold')

ax1.axis('off')

plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='NDVI')

 

# NDVI histogram

ax2 = plt.subplot(1, 2, 2)

ax2.hist(ndvi.flatten(), bins=100, color='green', alpha=0.7, edgecolor='black')

ax2.axvline(ndvi_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {ndvi_mean:.3f}')

ax2.axvline(ndvi_median, color='blue', linestyle='--', linewidth=2, label=f'Median: {ndvi_median:.3f}')

ax2.set_xlabel('NDVI Value', fontsize=11, fontweight='bold')

ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')

ax2.set_title('NDVI Distribution', fontsize=12, fontweight='bold')

ax2.legend()

ax2.grid(True, alpha=0.3)

 

plt.suptitle(f'NDVI Analysis - Leaves.raw\nRange: [{ndvi_min:.3f}, {ndvi_max:.3f}] | Timestamp: {timestamp}',

             fontsize=13, fontweight='bold')

plt.tight_layout()

 

out_ndvi = os.path.join(script_dir, f'Leaves_NDVI_{timestamp}.png')

plt.savefig(out_ndvi, dpi=150, bbox_inches='tight')

plt.close()

 

print(f"[{get_timestamp()}] Saved: {out_ndvi}")

 

print(f"\n[{get_timestamp()}] Processing complete!")

print("\nSPECTRAL CORRECTION SUMMARY:")

print("="*80)

print("Applied cross-talk correction coefficients:")

print("  R_corrected = R - 0.25×G - 0.5×IR  (removes green + IR contamination)")

print("  G_corrected = G - 0.3×B - 0.3×IR   (removes blue + IR contamination)")

print("  B_corrected = B - 0.15×R - 0.15×G - 0.4×IR  (removes red + green + IR)")

print("="*80)

print("\nNDVI RESULTS:")

print(f"  Formula: NDVI = (IR - R_corrected) / (IR + R_corrected)")

print(f"  Mean: {ndvi_mean:.3f}")

print(f"  Median: {ndvi_median:.3f}")

print(f"  Range: [{ndvi_min:.3f}, {ndvi_max:.3f}]")

print("="*80)

print("\nNote: Spectral response curves from Mira220_DS000642_9-00.pdf datasheet")

print("were used as reference for cross-talk correction coefficients.")

print("="*80)