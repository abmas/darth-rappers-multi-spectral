import cv2
import numpy as np
import tifffile as tiff


MIRA_PATH = "03_color_corrected_Bilinear_Debayering.png"
MAPIR_PATH = "Vijay_0315_leaves_tif.tif"
OUT_TIFF = "Vijay_0315_leaves_tif_aligned_to_mira.tif"
OUT_OVERLAY = "Vijay_0315_leaves_tif_aligned_overlay.png"
OUT_MASK = "Vijay_0315_leaves_tif_aligned_mask.png"

ARUCO_DICT = cv2.aruco.DICT_4X4_1000
MARKER_ID = 830


def normalize_to_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image

    image_f = image.astype(np.float32)
    span = max(float(image_f.max() - image_f.min()), 1e-6)
    return ((image_f - image_f.min()) * 255.0 / span).astype(np.uint8)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def load_mira_png(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return image


def load_mapir_tiff(path: str) -> np.ndarray:
    image = tiff.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return image


def detect_marker_corners(gray: np.ndarray) -> np.ndarray:
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        raise RuntimeError("No ArUco markers detected")

    ids = ids.flatten()
    for marker_corners, marker_id in zip(corners, ids):
        if marker_id == MARKER_ID:
            return marker_corners.reshape(4, 2).astype(np.float32)

    raise RuntimeError(f"Marker {MARKER_ID} not found")


def order_corners(corners: np.ndarray) -> np.ndarray:
    pts = corners.astype(np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def make_overlay(mira_image: np.ndarray, aligned_mapir_u8: np.ndarray) -> np.ndarray:
    mira_u8 = normalize_to_u8(mira_image)
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
    mira_image = load_mira_png(MIRA_PATH)
    mapir_image = load_mapir_tiff(MAPIR_PATH)

    mira_gray = to_gray(normalize_to_u8(mira_image))
    mapir_gray = to_gray(normalize_to_u8(mapir_image))

    mira_corners = order_corners(detect_marker_corners(mira_gray))
    mapir_corners = order_corners(detect_marker_corners(mapir_gray))

    homography, _ = cv2.findHomography(mapir_corners, mira_corners)
    if homography is None:
        raise RuntimeError("Failed to compute homography from ArUco corners")

    out_h, out_w = mira_gray.shape
    aligned_mapir = cv2.warpPerspective(
        mapir_image,
        homography,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
    )

    coverage_mask = cv2.warpPerspective(
        np.full(mapir_gray.shape, 255, dtype=np.uint8),
        homography,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
    )

    aligned_mapir_u8 = normalize_to_u8(aligned_mapir)
    overlay = make_overlay(mira_image, aligned_mapir_u8)

    tiff.imwrite(OUT_TIFF, aligned_mapir)
    cv2.imwrite(OUT_OVERLAY, overlay)
    cv2.imwrite(OUT_MASK, coverage_mask)

    print("MIRA size:", (out_w, out_h))
    print("MAPIR source size:", (mapir_image.shape[1], mapir_image.shape[0]))
    print("Homography:\n", homography)
    print("Saved:", OUT_TIFF)
    print("Saved:", OUT_OVERLAY)
    print("Saved:", OUT_MASK)


if __name__ == "__main__":
    main()
