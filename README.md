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


## Function Descriptions 

### 1-) reconstruction.h

**cv::Mat estimateFundamentalMatrix(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2)**  
**Inputs:**  
- pts1: 2D points from the left image  
- pts2: 2D points from the right image
   
**Output:**  
- Fundamental matrix F
    
**Purpose:**  
- Estimates the fundamental matrix using the normalized 8-point algorithm.

---

**cv::Mat estimateFundamentalMatrixRANSAC(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, std::vector<uchar>& inlierMask, int iterations, float threshold)**  
**Inputs:**  
- pts1, pts2: matched 2D points  
- iterations: number of RANSAC iterations  
- threshold: Sampson error threshold
  
**Outputs:**  
- Robust fundamental matrix F  
- inlierMask: binary mask of inliers
   
**Purpose:**  
- Computes the fundamental matrix robustly using RANSAC and Sampson error.

---

**void normalizePoints(const std::vector<cv::Point2f>& pts, std::vector<cv::Point2f>& normPts, cv::Mat& T)**  
**Inputs:**  
- pts: original 2D points
   
**Outputs:**  
- normPts: normalized points  
- T: normalization transformation matrix
  
**Purpose:**  
- Normalizes input points to improve numerical stability.

---

**cv::Mat recoverPoseCustom(const cv::Mat& E, const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, const cv::Mat& K, cv::Mat& R, cv::Mat& t)**  
**Inputs:**  
- E: essential matrix  
- pts1, pts2: matched 2D points  
- K: camera intrinsics
   
**Outputs:**  
- R: rotation matrix  
- t: translation vector
  
**Purpose:**  
- Recovers camera pose from the essential matrix using a cheirality check.

---

**void computeStereoRectification(const cv::Mat& K1, const cv::Mat& D1, const cv::Mat& K2, const cv::Mat& D2, const cv::Size& imageSize, const cv::Mat& R, const cv::Mat& T, cv::Mat& R1, cv::Mat& R2, cv::Mat& P1, cv::Mat& P2, cv::Mat& Q)**  
**Inputs:**  
- K1, K2: camera intrinsics  
- D1, D2: distortion coefficients  
- imageSize: size of the stereo images  
- R, T: rotation and translation between cameras
  
**Outputs:**  
- R1, R2: rectification rotations  
- P1, P2: projection matrices  
- Q: disparity-to-depth mapping matrix
  
**Purpose:**  
- Computes stereo rectification and projection matrices for both cameras.

---

**void computeRectificationMap(const cv::Mat& K, const cv::Mat& R, const cv::Mat& P, const cv::Size& imageSize, cv::Mat& mapX, cv::Mat& mapY)**  
**Inputs:**  
- K: camera intrinsics  
- R: rotation matrix  
- P: projection matrix  
- imageSize: image size
  
**Outputs:**  
- mapX, mapY: remap matrices
  
**Purpose:**  
- Computes remapping coordinates for image rectification.

---

**void remapBilinear(const cv::Mat& src, cv::Mat& dst, const cv::Mat& mapX, const cv::Mat& mapY)**  
**Inputs:**  
- src: source image  
- mapX, mapY: remapping matrices
  
**Output:**  
- dst: remapped image
  
**Purpose:**  
- Applies bilinear interpolation using custom remap maps.

---

**template <typename T> T clampValue(T val, T low, T high)**  
**Inputs:**  
- val: value to clamp  
- low: minimum value  
- high: maximum value
  
**Output:**  
- Clamped value
  
**Purpose:**  
- Utility function to keep a value within bounds.

### 2-) SemiGlobalMatching.h

**SemiGlobalMatching::SemiGlobalMatching(const cv::Mat& left, const cv::Mat& right, int max_disp, int census_size, int penalty_1, int penalty_2, float lr_threshold)**  
**Inputs:**  
- left, right: rectified stereo images  
- max_disp: maximum disparity  
- census_size: size of the census transform window  
- penalty_1, penalty_2: smoothness penalties  
- lr_threshold: left-right consistency threshold
   
**Purpose:**  
- Initializes the Semi-Global Matching pipeline and allocates memory.

---

**void SemiGlobalMatching::SGM_process()**  
**Purpose:**  
- Runs the entire SGM pipeline step-by-step: census, cost computation, aggregation, WTA, consistency check, subpixel refinement, and filtering.

---

**void SemiGlobalMatching::compute_census_transform()**  
**Purpose:**  
- Computes census bitstring representation for each pixel in left and right images.

---

**uint64_t SemiGlobalMatching::compute_census_at(const cv::Mat& img, int x, int y) const**  
**Inputs:**  
- img: grayscale image  
- x, y: pixel location
  
**Output:**  
- 64-bit census value
  
**Purpose:**  
- Computes census transform at a specific pixel location.

---

**int SemiGlobalMatching::hamming_distance(uint64_t a, uint64_t b) const**  
**Inputs:**  
- a, b: 64-bit census descriptors
  
**Output:**  
- Hamming distance between a and b
  
**Purpose:**  
- Measures bitwise difference between two census values.

---

**void SemiGlobalMatching::compute_matching_cost()**  
**Purpose:**  
- Computes cost volume using Hamming distance between left and right census codes.

---

**void SemiGlobalMatching::aggregate_costs()**  
**Purpose:**  
- Aggregates matching costs along 8 directions using SGM rules.

---

**void SemiGlobalMatching::aggregate_path(int start_x, int start_y, int dx, int dy)**  
**Inputs:**  
- start_x, start_y: starting pixel  
- dx, dy: direction of aggregation
    
**Purpose:**  
- Aggregates matching costs along a single path using penalties.

---

**void SemiGlobalMatching::compute_disparity_WTA()**  
**Purpose:**  
- Computes disparity map using the Winner-Takes-All strategy.

---

**void SemiGlobalMatching::compute_right_disparity()**  
**Purpose:**  
- Computes disparity map from the right view for left-right consistency check.

---

**void SemiGlobalMatching::left_right_consistency_check()**  
**Purpose:**  
- Invalidates disparity values if they differ too much between left and right maps.

---

**void SemiGlobalMatching::sub_pixel_refinement()**  
**Purpose:**  
- Refines disparity values to subpixel accuracy using quadratic interpolation.

---

**void SemiGlobalMatching::median_filter()**  
**Purpose:**  
- Applies a median filter to reduce noise in the disparity map.

---

**bool SemiGlobalMatching::is_valid_pixel(int x, int y) const**  
**Inputs:**  
- x, y: pixel coordinates
  
**Output:**  
- true if pixel is inside image bounds
  
**Purpose:**  
- Checks if a pixel is valid for processing.

---

**cv::Mat SemiGlobalMatching::get_disparity() const**  
**Output:**  
- 8-bit disparity map (left view)
   
**Purpose:**  
- Returns the final disparity map.

---

**cv::Mat SemiGlobalMatching::get_disparity_float() const**  
**Output:**  
- Float disparity map with subpixel accuracy
  
**Purpose:**  
- Returns the subpixel-refined disparity map.

---

**cv::Mat SemiGlobalMatching::get_disparity_left_right() const**  
**Output:**  
- Float disparity map from the right view
  
**Purpose:**  
- Returns the right-view disparity map for consistency checking.

### 3-) pointcloud_utils.h

**void vectorizedReprojectTo3D(const cv::Mat& disp, cv::Mat& depthMap, const cv::Mat& Q)**  
**Inputs:**  
- disp: input disparity map (CV_32F)  
- Q: 4x4 reprojection matrix from stereo calibration
  
**Output:**  
- depthMap: output 3D point map (CV_32FC3)
  
**Purpose:**  
- Projects disparity map into 3D space using matrix multiplication and vectorized operations for better performance.

---

**void manualReprojectTo3D(const cv::Mat& disp, cv::Mat& depthMap, const cv::Mat& Q, const bool vectorized = false)**  
**Inputs:**  
- disp: input disparity map (CV_32F)  
- Q: reprojection matrix  
- vectorized: optional flag to choose between manual loop or vectorized version
    
**Output:**  
- depthMap: output 3D point map (CV_32FC3)
  
**Purpose:**  
- Computes 3D coordinates for each pixel using the disparity and Q matrix. Supports both manual and vectorized backends.

---

**void writePointCloudPLY(const cv::Mat& disparityMap, const cv::Mat& depthMap, const cv::Mat& leftImage, const std::string& filename)**  
**Inputs:**  
- disparityMap: disparity values (CV_32F)  
- depthMap: 3D points (CV_32FC3)  
- leftImage: original left color image (CV_8UC3)  
- filename: output `.ply` file path
  
**Purpose:**  
- Exports a colored 3D point cloud as a PLY file using disparity and depth maps.

---

**void writeColorMeshPLY(const cv::Mat& dispFloat, const cv::Mat& depthMap, const cv::Mat& colorImage, const std::string& filename, float max_z_diff = 5.0f)**  
**Inputs:**  
- dispFloat: float disparity map  
- depthMap: reprojected 3D map (CV_32FC3)  
- colorImage: RGB texture source image  
- filename: output mesh filename (PLY)  
- max_z_diff: optional depth discontinuity threshold
  
**Purpose:**  
- Creates a colored triangular mesh from the depth map and exports it in PLY format. Neighboring points are connected if their Z-difference is within the threshold.

## Installation & Usage

  ### 1-) Dependencies
  - OpenCV 4.x
  - C++17-compatible compiler (e.g., g++)
  - pkg-config (to install: `sudo apt install pkg-config`)
  - OpenMP (to install: `sudo apt install libomp-dev`)
    
  ### 2-) Running the Custom Pipeline
  - open terminal at project root, navigate to the `source` folder with "cd source" command, and type: 
  - g++ -std=c++17 custom.cpp reconstruction.cpp  SemiGlobalMatching.cpp -o app  $(pkg-config --cflags --libs opencv4) -O3 -march=native -funroll-loops -fopenmp
  -> For run with sift:  type `./app sift`
  -> For run with brisk: type `./app brisk`

   ### 3-) Running the OpenCV Pipeline
  - open terminal at project root, navigate to the `source` folder with "cd source" command, and type: 
  - g++ -std=c++17 opencv.cpp reconstruction.cpp  SemiGlobalMatching.cpp -o app  $(pkg-config --cflags --libs opencv4) -O3 -march=native -funroll-loops -fopenmp
  -> For run with sift:  type `./app sift`
  -> For run with brisk: type `./app brisk`




