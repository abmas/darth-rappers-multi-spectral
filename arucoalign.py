import cv2
import tifffile as tiff
import numpy as np

# ---------- paths ----------
IMG1_PATH = "./Vijay_0315_leaves_tif.tif"   # reference
IMG2_PATH = "./20260315.png.png"            # to align

OUT_OVERLAY = "./alignment_overlay.png"
OUT_SIDE = "./alignment_side_by_side.png"

# ---------- marker ----------
ARUCO_DICT = cv2.aruco.DICT_4X4_1000
MARKER_ID = 830


def load_gray(path):
    if path.lower().endswith((".tif", ".tiff")):
        img = tiff.imread(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(path)

    # normalize to 8-bit
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = 255 * (img - img.min()) / max(img.max() - img.min(), 1e-6)
        img = img.astype(np.uint8)

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    return gray


def detect_marker(gray):
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None

    ids = ids.flatten()
    for c, i in zip(corners, ids):
        if i == MARKER_ID:
            return c.reshape(4, 2).astype(np.float32)
    return None


def order_pts(pts):
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    out = np.zeros((4, 2), dtype=np.float32)
    out[0] = pts[np.argmin(s)]   # top-left
    out[2] = pts[np.argmax(s)]   # bottom-right
    out[1] = pts[np.argmin(d)]   # top-right
    out[3] = pts[np.argmax(d)]   # bottom-left
    return out


# 1) load images
gray1 = load_gray(IMG1_PATH)
gray2 = load_gray(IMG2_PATH)

# 2) FIX: mirror second image BEFORE processing
gray2 = cv2.flip(gray2, 1)   # horizontal flip

# 3) detect marker corners
pts1 = detect_marker(gray1)
pts2 = detect_marker(gray2)

if pts1 is None:
    raise RuntimeError("Marker not found in first image")
if pts2 is None:
    raise RuntimeError("Marker not found in second image")

pts1 = order_pts(pts1)
pts2 = order_pts(pts2)

# 4) compute homography
H, _ = cv2.findHomography(pts2, pts1)

# 5) warp second image
aligned2 = cv2.warpPerspective(gray2, H, (gray1.shape[1], gray1.shape[0]))

# 6) visualization

# overlay (green = reference, magenta = aligned)
overlay = np.zeros((gray1.shape[0], gray1.shape[1], 3), dtype=np.uint8)
overlay[:, :, 1] = gray1
overlay[:, :, 0] = aligned2
overlay[:, :, 2] = aligned2

# side-by-side
before = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
side = np.hstack([gray1, before, aligned2])

cv2.imwrite(OUT_OVERLAY, overlay)
cv2.imwrite(OUT_SIDE, side)

print("Homography:\n", H)
print("Saved:", OUT_OVERLAY)
print("Saved:", OUT_SIDE)