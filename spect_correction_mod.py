#!/usr/bin/env python3
"""
DGK Color Card Calibration for MIRA220
--------------------------------------
This version:

✔ Loads Cal_leaves.raw
✔ Loads dark_vijay.raw and white.raw
✔ Performs black/white calibration
✔ Normalizes to 0–1
✔ Debayers with 4×4 per-block horizontal flip
✔ Upscales for clicking
✔ User clicks 18 patches in fixed order
✔ Uses built-in CMYK table
✔ Converts CMYK → sRGB → linear RGB
✔ Solves 4×4 spectral correction matrix
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FILES & SIZE
# ============================================================
RAW_FILE   = "Cal_leaves.raw"
DARK_FILE  = "raj_morning_dark.raw"
WHITE_FILE = "white.raw"

WIDTH  = 1600
HEIGHT = 1400

# ============================================================
# FIXED CMYK TABLE (18 patches)
# ============================================================
CMYK = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 35],
    [0, 0, 0, 50],
    [0, 0, 0, 65],
    [0, 0, 0, 80],
    [75, 68, 67, 90],

    [0, 100, 100, 0],
    [0, 0, 100, 0],
    [63, 0, 100, 0],
    [100, 0, 0, 0],
    [100, 80, 0, 60],
    [0, 100, 0, 0],

    [23, 93, 78, 13],
    [15, 57, 100, 2],
    [84, 33, 27, 1],
    [26, 80, 10, 0],
    [22, 38, 44, 1],
    [28, 42, 50, 2]
], dtype=np.float32)

# ============================================================
# BLACK/WHITE CALIBRATION
# ============================================================
def calibrate(raw, dark, white):
    num = raw - dark
    den = white - dark
    den[den == 0] = 1
    out = num / den
    return np.clip(out, 0.0, 1.0)

# ============================================================
# DEBAYER WITH PER-BLOCK HORIZONTAL FLIP
# ============================================================
def debayer_4x4_blockflip(bayer, h, w):
    H = h // 4
    W = w // 4

    R  = np.zeros((H, W), np.float32)
    G  = np.zeros((H, W), np.float32)
    B  = np.zeros((H, W), np.float32)
    IR = np.zeros((H, W), np.float32)

    for i in range(H):
        for j in range(W):
            block = bayer[i*4:(i+1)*4, j*4:(j+1)*4]
            block = block[:, ::-1]   # per-block horizontal flip

            R[i,j]  = (block[0,2] + block[2,0]) / 2.0
            B[i,j]  = (block[0,0] + block[2,2]) / 2.0

            G[i,j]  = (
                block[0,1]+block[0,3]+block[1,0]+block[1,2]+
                block[2,1]+block[2,3]+block[3,0]+block[3,2]
            ) / 8.0

            IR[i,j] = (block[1,1]+block[1,3]+block[3,1]+block[3,3]) / 4.0

    return R, G, B, IR

# ============================================================
# sRGB → linear
# ============================================================
def srgb_to_linear(s):
    return np.where(
        s <= 0.04045,
        s / 12.92,
        ((s + 0.055) / 1.055) ** 2.4
    )

# ============================================================
# Solve 4×4 spectral matrix
# ============================================================
def solve_matrix(measured, reference_linear):
    A_T, _, _, _ = np.linalg.lstsq(measured, reference_linear, rcond=None)
    A = A_T.T  # 3×4

    M = np.zeros((4,4), np.float32)
    M[:3,:] = A
    M[3,:] = [0,0,0,1]
    return M

# ============================================================
# MAIN
# ============================================================
def main():

    print("\nLoading RAW, DARK, WHITE...")
    raw   = np.fromfile(RAW_FILE,   dtype=np.uint16).reshape((HEIGHT, WIDTH)).astype(np.float32)
    dark  = np.fromfile(DARK_FILE,  dtype=np.uint16).reshape((HEIGHT, WIDTH)).astype(np.float32)
    white = np.fromfile(WHITE_FILE, dtype=np.uint16).reshape((HEIGHT, WIDTH)).astype(np.float32)

    # ----------------------------
    # 🔹 BLACK/WHITE CALIBRATION
    # ----------------------------
    print("Calibrating...")
    cal = calibrate(raw, dark, white)   # final: [0,1]

    # ----------------------------
    # 🔹 DEBAYER
    # ----------------------------
    print("Debayering...")
    R0, G0, B0, IR0 = debayer_4x4_blockflip(cal, HEIGHT, WIDTH)

    # Upscale for clicking (visual only)
    R_u  = R0.repeat(4,axis=0).repeat(4,axis=1)
    G_u  = G0.repeat(4,axis=0).repeat(4,axis=1)
    B_u  = B0.repeat(4,axis=0).repeat(4,axis=1)
    IR_u = IR0.repeat(4,axis=0).repeat(4,axis=1)

    rgb = np.stack([R_u, G_u, B_u], axis=-1)
    rgb_disp = np.clip(rgb / np.percentile(rgb, 99.5), 0, 1)

    print("\nClick EXACTLY 18 patches in correct order.")
    plt.figure(figsize=(10,7))
    plt.imshow(rgb_disp)
    pts = plt.ginput(18)
    plt.close()

    if len(pts) != 18:
        print("❌ Error: 18 clicks required.")
        return

    # ----------------------------
    # Extract patch means
    # ----------------------------
    measured = []
    rad = 12

    print("\nExtracting RGBIR values...")
    for (x,y) in pts:
        x = int(x); y = int(y)
        Rv = np.mean(R_u[y-rad:y+rad, x-rad:x+rad])
        Gv = np.mean(G_u[y-rad:y+rad, x-rad:x+rad])
        Bv = np.mean(B_u[y-rad:y+rad, x-rad:x+rad])
        IRv= np.mean(IR_u[y-rad:y+rad, x-rad:x+rad])
        measured.append([Rv, Gv, Bv, IRv])

    measured = np.array(measured)
    print("\nMeasured (calibrated & normalized) RGBIR:\n")
    print(measured)

    # ----------------------------
    # Convert CMYK → reference RGB (linear)
    # ----------------------------
    C = CMYK[:,0] / 100.0
    Mv= CMYK[:,1] / 100.0
    Y = CMYK[:,2] / 100.0
    K = CMYK[:,3] / 100.0

    R_srgb = (1 - C)*(1-K)
    G_srgb = (1 - Mv)*(1-K)
    B_srgb = (1 - Y)*(1-K)

    srgb = np.stack([R_srgb, G_srgb, B_srgb], axis=1)
    reference_linear = srgb_to_linear(srgb)

    # ----------------------------
    # SOLVE 4×4 MATRIX
    # ----------------------------
    M = solve_matrix(measured, reference_linear)

    print("\n==============================================")
    print("       NEW 4×4 SPECTRAL CORRECTION MATRIX")
    print("==============================================\n")
    print(M)
    print("\n==============================================\n")


if __name__ == "__main__":
    main()
