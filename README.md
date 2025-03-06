# RBE549-AutoCalib

## Project Overview
This project implements a robust and flexible **camera calibration technique** based on Zhengyou Zhang's method. The calibration process involves estimating the camera's intrinsic and extrinsic parameters, as well as distortion coefficients, using a planar checkerboard pattern. The method includes two main stages:
1. **Closed-form estimation** of intrinsic and extrinsic parameters using homographies.
2. **Non-linear optimization** to refine intrinsic parameters and distortion coefficients by minimizing reprojection error.

The calibrated parameters enable accurate undistortion and rectification of images, which are essential for 3D computer vision tasks.

---

## Key Features
- **Intrinsic Matrix Estimation:** Computes focal lengths, principal point, and skew coefficient.
- **Extrinsic Parameters Estimation:** Determines rotation and translation of the camera.
- **Distortion Coefficients:** Models radial distortion for accurate undistortion.
- **Reprojection Error Minimization:** Refines parameters using Levenberg-Marquardt optimization.
- **Visualization:** Generates undistorted images with reprojected corner points.

---

## Directory Structure
The project directory is organized as follows:

```
skhobragade_hw1
├── Calibration_Imgs
│   ├── IMG_20170209_042606.jpg
│   ├── IMG_20170209_042608.jpg
├── Output
│   ├── Detected_Chessboard_Corners
│   │   ├── 10.jpg
│   │   ├── 11.jpg
│   └── Reprojected_Corners
│       ├── reprojected_10.jpg
│       ├── reprojected_11.jpg
├── README.md
├── Report.pdf
└── Wrapper.py

```
The directory Calibration_Imgs must contain the images for calibration. The output after running the Wrapper.py script will be stored in Output directory. The Calibration_Imgs directory and Wrapper.py script should be inside the same directory.


## Running the Wrapper.py Script

### To run the code navigate to "/skhobragade_hw1" location first

To run the script, enter the below command in the terminal after navigating to the above mentioned directory.



```bash
python Wrapper.py
```

