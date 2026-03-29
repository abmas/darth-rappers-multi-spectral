import cv2
import tifffile as tiff
import numpy as np

# Load 3-channel TIFF
img = tiff.imread("./Vijay_0315_leaves_tif.tif")

# Convert 16-bit TIFF to 8-bit
img8 = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

# Convert to grayscale
gray = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)

# Optional: crop around the marker area for better reliability
crop = gray[100:900, 500:1800]

# Use the correct dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

corners, ids, rejected = detector.detectMarkers(crop)

print("Detected IDs:", ids)

# Draw result
vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
if ids is not None:
    cv2.aruco.drawDetectedMarkers(vis, corners, ids)

cv2.imwrite("detected_marker.png", vis)