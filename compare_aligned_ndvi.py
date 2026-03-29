import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MIRA_NDVI_PATH = "03_color_corrected_Bilinear_Debayering_ndvi.npy"
MAPIR_NDVI_PATH = "Vijay_0315_leaves_tif_aligned_to_mira_ndvi.npy"
MIRA_MASK_PATH = "03_color_corrected_Bilinear_Debayering.png"
MAPIR_MASK_PATH = "Vijay_0315_leaves_tif_aligned_mask.png"

VEG_THRESHOLD = 0.4

OUT_DIFF_NPY = "ndvi_difference_mira_minus_mapir.npy"
OUT_DIFF_HEATMAP = "ndvi_difference_mira_minus_mapir_heatmap.png"
OUT_OVERLAP_MASK = "ndvi_overlap_mask.png"
OUT_VEG_MASK = "ndvi_mapir_vegetation_mask_gt_0_4.png"


def load_png_mask(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, 3] > 0
    if image.ndim == 3:
        return np.any(image > 0, axis=2)
    return image > 0


def save_mask(mask: np.ndarray, path: str) -> None:
    cv2.imwrite(path, (mask.astype(np.uint8) * 255))


def main() -> None:
    mira_ndvi = np.load(MIRA_NDVI_PATH)
    mapir_ndvi = np.load(MAPIR_NDVI_PATH)

    mira_valid = load_png_mask(MIRA_MASK_PATH)
    mapir_valid = cv2.imread(MAPIR_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if mapir_valid is None:
        raise FileNotFoundError(MAPIR_MASK_PATH)
    mapir_valid = mapir_valid > 0

    overlap = mira_valid & mapir_valid & np.isfinite(mira_ndvi) & np.isfinite(mapir_ndvi)
    veg_mask = overlap & (mapir_ndvi > VEG_THRESHOLD)

    diff = np.full_like(mira_ndvi, np.nan, dtype=np.float32)
    diff[overlap] = mira_ndvi[overlap] - mapir_ndvi[overlap]

    np.save(OUT_DIFF_NPY, diff)
    save_mask(overlap, OUT_OVERLAP_MASK)
    save_mask(veg_mask, OUT_VEG_MASK)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(diff, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("NDVI Difference: MIRA - MAPIR")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NDVI delta")
    fig.tight_layout()
    fig.savefig(OUT_DIFF_HEATMAP, dpi=150, bbox_inches="tight")
    plt.close(fig)

    def metrics(mask: np.ndarray, label: str) -> None:
        vals_mira = mira_ndvi[mask]
        vals_mapir = mapir_ndvi[mask]
        vals_diff = vals_mira - vals_mapir
        mae = np.mean(np.abs(vals_diff))
        rmse = np.sqrt(np.mean(vals_diff ** 2))
        corr = np.corrcoef(vals_mira, vals_mapir)[0, 1] if vals_mira.size > 1 else np.nan
        print(label)
        print("  pixels:", int(mask.sum()))
        print("  mira_mean:", float(np.mean(vals_mira)))
        print("  mapir_mean:", float(np.mean(vals_mapir)))
        print("  diff_mean:", float(np.mean(vals_diff)))
        print("  mae:", float(mae))
        print("  rmse:", float(rmse))
        print("  corr:", float(corr))

    metrics(overlap, "all_overlap")
    metrics(veg_mask, "mapir_vegetation_gt_0.4")
    print("Saved:", OUT_DIFF_NPY)
    print("Saved:", OUT_DIFF_HEATMAP)
    print("Saved:", OUT_OVERLAP_MASK)
    print("Saved:", OUT_VEG_MASK)


if __name__ == "__main__":
    main()
