#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include <opencv2/core.hpp>
#include <vector>

// Clamp function (used internally)
template <typename T>
T clampValue(T val, T low, T high);

// Bilinear remapping
void remapBilinear(const cv::Mat& src, cv::Mat& dst, const cv::Mat& mapX, const cv::Mat& mapY);

// Compute rectification maps manually
void computeRectificationMap(
    const cv::Mat& K, const cv::Mat& R, const cv::Mat& P,
    const cv::Size& imageSize, cv::Mat& mapX, cv::Mat& mapY);

// Compute stereo rectification manually (returns R1, R2, P1, P2, Q)
void computeStereoRectification(
    const cv::Mat& K1, const cv::Mat& D1,
    const cv::Mat& K2, const cv::Mat& D2,
    const cv::Size& imageSize,
    const cv::Mat& R, const cv::Mat& T,
    cv::Mat& R1, cv::Mat& R2,
    cv::Mat& P1, cv::Mat& P2,
    cv::Mat& Q);

// Normalize image points for F estimation
void normalizePoints(const std::vector<cv::Point2f>& pts,
                     std::vector<cv::Point2f>& normPts,
                     cv::Mat& T);

// Estimate Fundamental Matrix using normalized 8-point algorithm
cv::Mat estimateFundamentalMatrix(const std::vector<cv::Point2f>& pts1,
                                  const std::vector<cv::Point2f>& pts2);

// Estimate Fundamental Matrix using custom RANSAC + Sampson error
cv::Mat estimateFundamentalMatrixRANSAC(const std::vector<cv::Point2f>& pts1,
                                        const std::vector<cv::Point2f>& pts2,
                                        std::vector<uchar>& inlierMask,
                                        int iterations = 2000,
                                        float threshold = 1.0f);

// Recover relative pose from Essential matrix (custom disambiguation)
cv::Mat recoverPoseCustom(const cv::Mat& E,
                          const std::vector<cv::Point2f>& pts1,
                          const std::vector<cv::Point2f>& pts2,
                          const cv::Mat& K,
                          cv::Mat& R,
                          cv::Mat& t);

#endif // RECONSTRUCTION_H
