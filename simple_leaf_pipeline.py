#!/usr/bin/env python3
"""
SIMPLE LEAF SPECTRAL PROCESSING PIPELINE
==========================================

This script takes a raw image from the MIRA 220 leaf camera and processes it
through two different methods to extract RGB colors and measure plant health (NDVI).

Think of it like a photo developing process:
  1. Raw photo (black & white sensor data)
  2. Color correction (remove shadows and brightness issues)
  3. Color balancing (adjust color tones)
  4. Plant health measurement (calculate how green/healthy the leaf is)

The script shows pictures at each step so you can see what happens!
"""

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Show interactive windows
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.colors import ListedColormap, BoundaryNorm

# ============================================================
# SETUP: File paths and settings (change these if needed)
# ============================================================

WIDTH, HEIGHT = 1600, 1400          # Image size: 1600 pixels wide, 1400 tall
RAW_FILE = "raj2.raw"              # The raw sensor data file
BLACK_FILE = "dark_vijay.raw"       # Calibration: dark photo (for shadow correction)
WHITE_FILE = "white.raw"            # Calibration: bright photo (for brightness correction)
OUTPUT_DIR = "output"               # Where to save the results
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SPECTRAL CORRECTION MATRIX
# ============================================================
# This is a "color recipe" that adjusts the R, G, B values to match
# the true colors of the leaf. It's like adding spices to a recipe!
#
M = np.array([
    [ 3.9170194, -1.1103847,  0.0875303, -3.7035956],
    [-4.4927955,  5.0224247, -2.4812708,  5.106834 ],
    [ 1.2343036, -3.7725825,  5.121925,  -1.1049197],
    [ 0.0,         0.0,         0.0,         2.0    ]
], dtype=np.float32)

# ============================================================
# FUNCTION 1: Read a raw image file
# ============================================================
def read_raw(filename, w, h):
    """
    Load a RAW image file.
    
    The file contains raw pixel data (just numbers).
    We read them and arrange them in a grid (h rows, w columns).
    
    Args:
        filename: name of the .raw file
        w: width (number of columns)
        h: height (number of rows)
    
    Returns:
        A 2D grid of numbers representing the raw image
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find file: {filename}")
    
    # Read all numbers from the file
    data = np.fromfile(filename, dtype=np.uint16)
    expected = w * h
    
    if data.size != expected:
        raise ValueError(f"File size mismatch. Expected {expected} pixels, got {data.size}")
    
    # Arrange into a grid (height rows, width columns)
    return data.reshape((h, w))


# ============================================================
# FUNCTION 2: Apply 4×4 block horizontal flip
# ============================================================
def apply_4x4_blockflip_to_bayer(bayer):
    """
    The MIRA 220 sensor has a special pattern, and we need to "flip" 
    each 4×4 block sideways to make it match what we expect.
    
    Think of it like a Rubik's cube move - we're rotating small squares.
    
    Args:
        bayer: The raw image grid
    
    Returns:
        The flipped image grid
    """
    flipped = bayer.copy()
    h, w = bayer.shape
    
    # Go through each 4×4 block
    for i in range(0, h, 4):
        for j in range(0, w, 4):
            block = flipped[i:i+4, j:j+4]
            if block.shape == (4, 4):
                # Flip each block left-right ([:, ::-1] means flip columns)
                flipped[i:i+4, j:j+4] = block[:, ::-1]
    
    return flipped


# ============================================================
# FUNCTION 3: Calibration - Remove shadows and adjust brightness
# ============================================================
def calibrate_bayer(raw, black, white, apply_white_flip=True):
    """
    This is like "white balance" in photography.
    
    We use two reference photos:
    - BLACK_FILE: A dark photo (captures sensor noise and shadows)
    - WHITE_FILE: A bright photo (captures max brightness)
    
    We subtract the dark values and divide by the brightness range
    to get a "cleaned up" image from 0 to 1.
    
    Args:
        raw: The raw sensor image
        black: Dark reference image
        white: Bright reference image
        apply_white_flip: Whether to flip the calibration images (should be True)
    
    Returns:
        Calibrated image with values between 0 and 1
    """
    if apply_white_flip:
        white = apply_4x4_blockflip_to_bayer(white)
        black = apply_4x4_blockflip_to_bayer(black)
    
    # Convert to floating point for math
    raw_f = raw.astype(np.float32)
    black_f = black.astype(np.float32)
    white_f = white.astype(np.float32)
    
    # Subtract dark values (remove shadows)
    numerator = raw_f - black_f
    
    # Divide by brightness range (normalize to 0-1)
    denominator = white_f - black_f
    denominator[denominator == 0] = 1  # Avoid division by zero
    
    # Clip values to stay between 0 and 1
    return np.clip(numerator / denominator, 0, 1)


# ============================================================
# FUNCTION 4: Extract R, G, B channels from Bayer mosaic
# ============================================================
def debayer_4x4_hflip(bayer_raw, HEIGHT, WIDTH):
    """
    The MIRA 220 sensor doesn't have separate R, G, B cameras.
    Instead, each pixel is one color (Red, Green, Blue, or Infrared).
    
    This function "de-mosaics" - it figures out what color each pixel
    should be by looking at neighboring pixels.
    
    Think of it like a paint-by-numbers: we fill in missing colors.
    
    Args:
        bayer_raw: The raw sensor image
        HEIGHT, WIDTH: Image dimensions
    
    Returns:
        R, G, B, IR: Four separate color channel images
    """
    # Create empty grids for each color
    R = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    G = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    B = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    IR = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    
    # The MIRA 220 has a 4×4 repeating pattern with a flip
    # This pattern tells us where each color is located
    PIXEL_LABELS = np.array([
        ["B", "G", "R", "G"],
        ["G", "IR", "G", "IR"],
        ["R", "G", "B", "G"],
        ["G", "IR", "G", "IR"]
    ])
    
    # Go through each 4×4 block
    for i in range(0, HEIGHT, 4):
        for j in range(0, WIDTH, 4):
            block = bayer_raw[i:i+4, j:j+4]
            if block.shape != (4, 4):
                continue
            
            # Flip the block horizontally (required for MIRA 220)
            block_flipped = block[:, ::-1]
            
            # Extract each color from the known positions
            for dy in range(4):
                for dx in range(4):
                    label = PIXEL_LABELS[dy, dx]
                    value = block_flipped[dy, dx]
                    
                    y = i + dy
                    x = j + dx
                    
                    if label == "R":
                        R[y, x] = value
                    elif label == "G":
                        G[y, x] = value
                    elif label == "B":
                        B[y, x] = value
                    elif label == "IR":
                        IR[y, x] = value
    
    return R, G, B, IR


# ============================================================
# FUNCTION 5: Fill in missing colors (interpolation)
# ============================================================
def bilinear_fill_iter(channel, iterations=2):
    """
    After extracting R, G, B, there are still "holes" where we
    don't have that color. This function fills the holes by averaging
    nearby pixels - like smoothing out a pixelated image.
    
    It runs multiple times (iterations) to fill all the gaps.
    
    Args:
        channel: One color channel (R, G, or B)
        iterations: How many times to repeat the smoothing
    
    Returns:
        Filled-in color channel (no more holes)
    """
    result = channel.copy()
    h, w = result.shape
    
    for iteration in range(iterations):
        temp = result.copy()
        
        # For each pixel, if it's empty (0), average its neighbors
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if result[i, j] == 0:
                    # Average the 4 neighbors (up, down, left, right)
                    neighbors = [
                        result[i-1, j],  # above
                        result[i+1, j],  # below
                        result[i, j-1],  # left
                        result[i, j+1]   # right
                    ]
                    valid = [n for n in neighbors if n > 0]
                    if valid:
                        temp[i, j] = np.mean(valid)
        
        result = temp
    
    return result


# ============================================================
# FUNCTION 6: Apply color correction (spectral matrix)
# ============================================================
def apply_spectral_matrix(R, G, B, IR, M):
    """
    This is the "color recipe" - we adjust the red, green, and blue
    values to match the true colors of the leaf.
    
    It's like adjusting the color settings on a TV:
    maybe you want more red, less blue, etc.
    
    Args:
        R, G, B, IR: The four color channels
        M: The correction matrix (recipe)
    
    Returns:
        R, G, B, IR after color correction
    """
    h, w = R.shape
    
    # Create output images
    R_out = np.zeros((h, w), dtype=np.float32)
    G_out = np.zeros((h, w), dtype=np.float32)
    B_out = np.zeros((h, w), dtype=np.float32)
    IR_out = np.zeros((h, w), dtype=np.float32)
    
    # For each pixel, apply the color recipe
    for i in range(h):
        for j in range(w):
            # Get the input values for this pixel
            input_vec = np.array([R[i, j], G[i, j], B[i, j], IR[i, j]], dtype=np.float32)
            
            # Apply the matrix: output = M @ input
            output_vec = M @ input_vec
            
            # Store the corrected values
            R_out[i, j] = output_vec[0]
            G_out[i, j] = output_vec[1]
            B_out[i, j] = output_vec[2]
            IR_out[i, j] = output_vec[3]
    
    # Clip values to stay between 0 and 1
    return (np.clip(R_out, 0, 1), np.clip(G_out, 0, 1), 
            np.clip(B_out, 0, 1), np.clip(IR_out, 0, 1))


# ============================================================
# FUNCTION 7: Calculate NDVI (plant health index)
# ============================================================
def compute_ndvi(R, IR):
    """
    NDVI = (IR - R) / (IR + R)
    
    This measures how "green" or healthy the leaf is:
    - Green/healthy leaves reflect a lot of infrared
    - Red/stressed leaves reflect less infrared
    
    The result is between -1 and 1:
    - -1: Dead (no life)
    - 0: Soil/concrete (not plant)
    - +1: Super healthy green plant
    
    Args:
        R: Red channel
        IR: Infrared channel
    
    Returns:
        NDVI image (plant health map)
    """
    # Avoid division by zero
    denominator = IR.astype(np.float32) + R.astype(np.float32)
    denominator[denominator == 0] = 1
    
    numerator = IR.astype(np.float32) - R.astype(np.float32)
    ndvi = numerator / denominator
    
    return np.clip(ndvi, -1, 1)


# ============================================================
# FUNCTION 8: Create pretty false-color map for NDVI
# ============================================================
def create_false_color_ndvi(ndvi):
    """
    Convert NDVI numbers into a pretty color image:
    - Red: Dead or stressed
    - Yellow/Orange: Medium health
    - Green: Healthy
    - Dark Green: Super healthy
    
    Args:
        ndvi: NDVI image
    
    Returns:
        RGB false-color image
    """
    # Define color scale
    colors = [
        [1.0, 0.0, 0.0],      # Red (dead)
        [1.0, 0.5, 0.0],      # Orange (stressed)
        [1.0, 1.0, 0.0],      # Yellow (OK)
        [0.5, 1.0, 0.5],      # Light green (healthy)
        [0.0, 0.7, 0.0],      # Green (very healthy)
        [0.0, 0.4, 0.0]       # Dark green (super healthy)
    ]
    
    # Define the NDVI ranges for each color
    bounds = [-1.0, -0.1, 0.0, 0.3, 0.5, 0.7, 1.0]
    
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Create an RGB version of the NDVI image using the colors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    rgb = sm.to_rgba(ndvi)[:, :, :3]
    
    return np.clip(rgb, 0, 1)


# ============================================================
# FUNCTION 9: Build RGB image from channels
# ============================================================
def build_rgb(R, G, B):
    """
    Combine the three color channels into one RGB image
    that can be displayed.
    
    Args:
        R, G, B: Individual color channels
    
    Returns:
        RGB image (can be displayed with plt.imshow)
    """
    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0, 1)


# ============================================================
# FUNCTION 10: Display and save images
# ============================================================
def plot_and_save(title, image, filename, is_ndvi=False):
    """
    Show an image on screen and save it to a file.
    
    Args:
        title: What to call the image
        image: The image to display
        filename: Where to save it
        is_ndvi: If True, use grayscale; otherwise use RGB
    """
    plt.figure(figsize=(10, 8))
    
    if is_ndvi:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Save the image
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    
    # Show on screen
    plt.show()


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    """
    This is where everything happens!
    """
    print("\n" + "="*60)
    print("SIMPLE LEAF SPECTRAL PROCESSING PIPELINE")
    print("="*60)
    
    # Ask user which method to use
    print("\nChoose a processing method:")
    print("  1. Full old pipeline (basic)")
    print("  2. Bilinear debayering (smooth colors)")
    
    choice = input("\nEnter 1 or 2: ").strip()
    if choice not in {"1", "2"}:
        print("Invalid choice!")
        return
    
    method_name = "Full_Old_Pipeline" if choice == "1" else "Bilinear_Debayering"
    
    print(f"\n→ Using: {method_name}")
    print("\nLoading files...")
    
    # Step 1: Load the raw images
    raw = read_raw(RAW_FILE, WIDTH, HEIGHT)
    black = read_raw(BLACK_FILE, WIDTH, HEIGHT)
    white = read_raw(WHITE_FILE, WIDTH, HEIGHT)
    print("✓ Files loaded")
    
    # Step 2: Calibration (remove shadows and adjust brightness)
    print("\nCalibrating (removing shadows and adjusting brightness)...")
    cal = calibrate_bayer(raw, black, white, apply_white_flip=True)
    
    # Step 3: Extract color channels (debayering)
    print("\nExtracting color channels...")
    R, G, B, IR = debayer_4x4_hflip(cal, HEIGHT, WIDTH)
    
    # Step 4: Fill in missing colors
    print("Filling in missing colors (interpolation)...")
    R_filled = bilinear_fill_iter(R, iterations=2)
    G_filled = bilinear_fill_iter(G, iterations=2)
    B_filled = bilinear_fill_iter(B, iterations=2)
    IR_filled = bilinear_fill_iter(IR, iterations=2)
    
    # Show individual R, G, B, IR channels in COLOR
    print("\nShowing individual color channels...")
    
    # Red channel - displayed as RED
    red_colored = build_rgb(R_filled, np.zeros_like(R_filled), np.zeros_like(R_filled))
    plot_and_save("Red Channel (R) - Red Only", 
                  red_colored, f"{OUTPUT_DIR}/01a_channel_R_{method_name}.png")
    
    # Green channel - displayed as GREEN
    green_colored = build_rgb(np.zeros_like(G_filled), G_filled, np.zeros_like(G_filled))
    plot_and_save("Green Channel (G) - Green Only", 
                  green_colored, f"{OUTPUT_DIR}/01b_channel_G_{method_name}.png")
    
    # Blue channel - displayed as BLUE
    blue_colored = build_rgb(np.zeros_like(B_filled), np.zeros_like(B_filled), B_filled)
    plot_and_save("Blue Channel (B) - Blue Only", 
                  blue_colored, f"{OUTPUT_DIR}/01c_channel_B_{method_name}.png")
    
    # Infrared channel - displayed as MAGENTA (Red + Blue)
    ir_colored = build_rgb(IR_filled, np.zeros_like(IR_filled), IR_filled)
    plot_and_save("Infrared Channel (IR) - Magenta", 
                  ir_colored, f"{OUTPUT_DIR}/01d_channel_IR_{method_name}.png")
    
    # Show the debayered result (BRIGHT AND VISIBLE!)
    rgb_debayered = build_rgb(R_filled, G_filled, B_filled)
    plot_and_save("Step 1: After Debayering (colors extracted and filled)", 
                  rgb_debayered, f"{OUTPUT_DIR}/02_debayered_{method_name}.png")
    
    # Step 5: Apply color correction (spectral matrix)
    print("\nApplying color correction...")
    R_corr, G_corr, B_corr, IR_corr = apply_spectral_matrix(
        R_filled, G_filled, B_filled, IR_filled, M
    )
    
    # Show the corrected result
    rgb_corrected = build_rgb(R_corr, G_corr, B_corr)
    plot_and_save("Step 2: After Color Correction (spectral balance)", 
                  rgb_corrected, f"{OUTPUT_DIR}/03_color_corrected_{method_name}.png")
    
    # Step 6: Calculate NDVI (plant health)
    print("\nCalculating NDVI (plant health index)...")
    ndvi = compute_ndvi(R_corr, IR_corr)
    
    # Create false-color version
    ndvi_falsecolor = create_false_color_ndvi(ndvi)
    plot_and_save("Step 3: NDVI False Color (red=dead, green=healthy)", 
                  ndvi_falsecolor, f"{OUTPUT_DIR}/04_NDVI_{method_name}.png")
    
    # Final summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"\nAll images have been saved to: {OUTPUT_DIR}/")
    print("\nYou should see 4 images on your screen:")
    print("  1. Debayered RGB (colors extracted - BRIGHT!)")
    print("  2. Color Corrected (colors balanced)")
    print("  3. NDVI False Color (plant health)")
    print("  4. Done!")
    print("\n✓ Done!\n")


# ============================================================
# RUN THE PROGRAM
# ============================================================
if __name__ == "__main__":
    main()
