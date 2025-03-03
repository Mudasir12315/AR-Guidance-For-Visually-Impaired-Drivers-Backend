import cv2
import numpy as np
import glob

# Define checkerboard dimensions (inner corners, not squares)
CHECKERBOARD = (7, 7)  # Adjust based on your checkerboard pattern
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points: (0,0,0), (1,0,0), (2,0,0), ... up to CHECKERBOARD
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load calibration chessboard_images (provide the path to your chessboard_images)
images = glob.glob('../chessboard_images_resized/*.jpeg')  # Replace with your directory
print(images)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(refined_corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, refined_corners, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save results
print("Camera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)

# Save calibration data for future use
np.save('../calibration_matrix/camera_matrix.npy', camera_matrix)
np.save('../calibration_matrix/dist_coeffs.npy', dist_coeffs)

# Undistort an example image
example_img = cv2.imread(images[0])
h, w = example_img.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
undistorted_img = cv2.undistort(example_img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# Display and save the undistorted image
cv2.imshow("Undistorted Image", undistorted_img)
cv2.imwrite('../calibration_matrix/undistorted_example.jpg', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()