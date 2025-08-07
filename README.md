# Stereo Reconstruction - Custom Pipeline

## Introduction

This project implements a complete stereo 3D reconstruction pipeline using both **OpenCV’s built-in methods** and **custom-developed algorithms**. The goal is to estimate a dense 3D structure of a static scene from a pair of rectified stereo images. 

The input data is taken from the **Middlebury 2014 Motorcycle-perfect** stereo dataset, with calibration parameters preloaded accordingly.

The pipeline supports multiple feature extractors (e.g., **SIFT** and **BRISK**) and can generate both **point clouds** and **3D meshes** from stereo image pairs. A key focus is on comparing OpenCV’s `StereoSGBM` with our **custom implementation** of the Semi-Global Matching (SGM) algorithm.

---

## Pipeline Overview

### 1. Feature Detection and Matching

- Keypoints and descriptors are computed using **SIFT** or **BRISK**.
- Descriptors are matched using **Brute-Force KNN Matching**.
- Matches are filtered using **Lowe’s ratio test** and descriptor **distance thresholding**.

### 2. Pose Estimation and Rectification

- The **fundamental matrix** is estimated using **RANSAC**.
- The **essential matrix** is computed and camera pose is recovered via a custom `recoverPoseCustom()` function.
- **Stereo rectification** is performed using custom mapping functions:  
  `computeStereoRectification`, `computeRectificationMap`, and `remapBilinear`.

### 3. Dense Disparity Map Generation

Two disparity computation approaches are supported:

#### a) OpenCV StereoSGBM  
Built-in OpenCV method for dense disparity estimation.

#### b) Custom SGM Implementation  
A fully custom 8-path Semi-Global Matching (SGM) pipeline:
- **Census Transform**
- **Hamming distance** for cost computation
- **Cost aggregation** in 8 directions
- **Winner-Takes-All (WTA)** disparity selection
- **Left-right consistency check**
- **Subpixel refinement**
- **Median filtering**

### 4. 3D Projection and Mesh Generation

- The disparity map is reprojected into 3D space using a scaled **Q matrix**.
- **Point clouds** are saved as `.ply` files.
- **Colored meshes** are generated and saved using a **custom mesh function**.

---

## Dataset

- Dataset: [Middlebury 2014 - Motorcycle-perfect](https://vision.middlebury.edu/stereo/data/scenes2014/)
