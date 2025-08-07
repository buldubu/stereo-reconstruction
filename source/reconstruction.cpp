#include "reconstruction.h"
#include <iostream>
#include <cmath>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

// Clamp utility
template <typename T>
T clampValue(T val, T low, T high) {
    return std::min(std::max(val, low), high);
}

// Bilinear remapping implementation
void remapBilinear(const cv::Mat& src, cv::Mat& dst, const cv::Mat& mapX, const cv::Mat& mapY) {
    dst.create(mapX.size(), src.type());
    int rows = mapX.rows, cols = mapX.cols, channels = src.channels();

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            float fx = mapX.at<float>(y, x);
            float fy = mapY.at<float>(y, x);

            int x0 = static_cast<int>(std::floor(fx));
            int y0 = static_cast<int>(std::floor(fy));
            int x1 = x0 + 1, y1 = y0 + 1;
            float dx = fx - x0, dy = fy - y0;

            for (int c = 0; c < channels; ++c) {
                float I00 = 0, I10 = 0, I01 = 0, I11 = 0;
                if (x0 >= 0 && y0 >= 0 && x0 < src.cols && y0 < src.rows)
                    I00 = src.at<cv::Vec3b>(y0, x0)[c];
                if (x1 >= 0 && y0 >= 0 && x1 < src.cols && y0 < src.rows)
                    I10 = src.at<cv::Vec3b>(y0, x1)[c];
                if (x0 >= 0 && y1 >= 0 && x0 < src.cols && y1 < src.rows)
                    I01 = src.at<cv::Vec3b>(y1, x0)[c];
                if (x1 >= 0 && y1 >= 0 && x1 < src.cols && y1 < src.rows)
                    I11 = src.at<cv::Vec3b>(y1, x1)[c];

                float interpolated =
                    (1 - dx) * (1 - dy) * I00 +
                    dx * (1 - dy) * I10 +
                    (1 - dx) * dy * I01 +
                    dx * dy * I11;

                interpolated = clampValue(interpolated, 0.0f, 255.0f);
                dst.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(interpolated + 0.5f);
            }
        }
    }
}

// Undistort and rectify mapping computation
void computeRectificationMap(
    const cv::Mat& K, const cv::Mat& R, const cv::Mat& P,
    const cv::Size& imageSize, cv::Mat& mapX, cv::Mat& mapY)
{
    mapX.create(imageSize, CV_32FC1);
    mapY.create(imageSize, CV_32FC1);
    cv::Mat Kinv = K.inv();

    for (int y = 0; y < imageSize.height; ++y) {
        for (int x = 0; x < imageSize.width; ++x) {
            cv::Mat pt = (cv::Mat_<double>(3, 1) << x, y, 1.0);
            cv::Mat norm = Kinv * pt;
            cv::Mat rect = R * norm;

            double X = rect.at<double>(0);
            double Y = rect.at<double>(1);
            double Z = rect.at<double>(2);

            double u = P.at<double>(0, 0) * X + P.at<double>(0, 1) * Y + P.at<double>(0, 2) * Z;
            double v = P.at<double>(1, 0) * X + P.at<double>(1, 1) * Y + P.at<double>(1, 2) * Z;
            double w = P.at<double>(2, 0) * X + P.at<double>(2, 1) * Y + P.at<double>(2, 2) * Z;

            mapX.at<float>(y, x) = static_cast<float>(u / w);
            mapY.at<float>(y, x) = static_cast<float>(v / w);
        }
    }
}

// Stereo rectification computation
void computeStereoRectification(
    const cv::Mat& K1, const cv::Mat& D1,
    const cv::Mat& K2, const cv::Mat& D2,
    const cv::Size& imageSize,
    const cv::Mat& R, const cv::Mat& T,
    cv::Mat& R1, cv::Mat& R2,
    cv::Mat& P1, cv::Mat& P2,
    cv::Mat& Q)
{
    cv::Mat c2 = -R.t() * T;
    cv::Mat v = c2 / cv::norm(c2);
    cv::Mat tmp = (cv::Mat_<double>(3, 1) << 0, 1, 0);
    if (std::abs(v.dot(tmp)) > 0.9)
        tmp = (cv::Mat_<double>(3, 1) << 1, 0, 0);

    cv::Mat x = v;
    cv::Mat y = tmp - x * (x.dot(tmp));
    y /= cv::norm(y);
    cv::Mat z = x.cross(y);
    cv::Mat Rrect(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i) {
        Rrect.at<double>(0, i) = x.at<double>(i);
        Rrect.at<double>(1, i) = y.at<double>(i);
        Rrect.at<double>(2, i) = z.at<double>(i);
    }

    R1 = Rrect;
    R2 = Rrect * R;

    cv::Mat K1_new = cv::getOptimalNewCameraMatrix(K1, D1, imageSize, 0, imageSize);
    cv::Mat K2_new = cv::getOptimalNewCameraMatrix(K2, D2, imageSize, 0, imageSize);

    double fx = (K1_new.at<double>(0, 0) + K2_new.at<double>(0, 0)) / 2.0;
    double fy = (K1_new.at<double>(1, 1) + K2_new.at<double>(1, 1)) / 2.0;
    double cx1 = K1_new.at<double>(0, 2);
    double cy1 = K1_new.at<double>(1, 2);
    double cx2 = K2_new.at<double>(0, 2);
    double cy2 = K2_new.at<double>(1, 2);

    cv::Mat T_new = Rrect * T;
    double Tx = std::abs(T_new.at<double>(0, 0));

    P1 = (cv::Mat_<double>(3, 4) << fx, 0, cx1, 0, 0, fy, cy1, 0, 0, 0, 1, 0);
    P2 = (cv::Mat_<double>(3, 4) << fx, 0, cx2, -fx * Tx, 0, fy, cy2, 0, 0, 0, 1, 0);

    Q = (cv::Mat_<double>(4, 4) <<
        1, 0, 0, -cx1,
        0, 1, 0, -cy1,
        0, 0, 0, fx,
        0, 0, 1.0 / Tx, -(cx1 - cx2) / Tx);
}

cv::Mat estimateFundamentalMatrix(const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2);

cv::Mat estimateFundamentalMatrixRANSAC(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    std::vector<uchar>& inlierMask,
    int iterations,
    float threshold)
{
    CV_Assert(pts1.size() == pts2.size() && pts1.size() >= 8);

    int N = pts1.size();
    cv::RNG rng;
    int bestInliers = 0;
    cv::Mat bestF;
    std::vector<uchar> bestMask;

    for (int it = 0; it < iterations; ++it) {
        // 1. Sample 8 unique indices
        std::set<int> indices;
        while (indices.size() < 8)
            indices.insert(rng.uniform(0, N));

        std::vector<cv::Point2f> sample1, sample2;
        for (int idx : indices) {
            sample1.push_back(pts1[idx]);
            sample2.push_back(pts2[idx]);
        }

        // 2. Estimate F
        cv::Mat F = estimateFundamentalMatrix(sample1, sample2);

        // 3. Count inliers using Sampson error
        std::vector<uchar> mask(N, 0);
        int inlierCount = 0;
        for (int i = 0; i < N; ++i) {
            const auto& p1 = pts1[i];
            const auto& p2 = pts2[i];
            cv::Mat x1 = (cv::Mat_<double>(3, 1) << p1.x, p1.y, 1.0);
            cv::Mat x2 = (cv::Mat_<double>(3, 1) << p2.x, p2.y, 1.0);

            cv::Mat Fx1 = F * x1;
            cv::Mat Ftx2 = F.t() * x2;
            double x2tFx1 = x2.dot(Fx1);

            double d = x2tFx1 * x2tFx1;
            double denom = Fx1.at<double>(0) * Fx1.at<double>(0) +
                Fx1.at<double>(1) * Fx1.at<double>(1) +
                Ftx2.at<double>(0) * Ftx2.at<double>(0) +
                Ftx2.at<double>(1) * Ftx2.at<double>(1);

            double err = d / denom;
            if (err < threshold) {
                mask[i] = 1;
                ++inlierCount;
            }
        }

        // 4. Keep best F
        if (inlierCount > bestInliers) {
            bestInliers = inlierCount;
            bestF = F.clone();
            bestMask = mask;
        }
    }

    // 5. Re-estimate F using all inliers
    std::vector<cv::Point2f> inliers1, inliers2;
    for (int i = 0; i < N; ++i) {
        if (bestMask[i]) {
            inliers1.push_back(pts1[i]);
            inliers2.push_back(pts2[i]);
        }
    }

    inlierMask = bestMask;
    return estimateFundamentalMatrix(inliers1, inliers2);
}


void normalizePoints(const std::vector<cv::Point2f>& pts,
    std::vector<cv::Point2f>& normPts,
    cv::Mat& T) {
    cv::Point2f centroid(0, 0);
    for (const auto& pt : pts) centroid += pt;
    centroid *= 1.0f / pts.size();

    double scale = 0;
    for (const auto& pt : pts) scale += cv::norm(pt - centroid);
    scale = std::sqrt(2.0) * pts.size() / scale;

    T = (cv::Mat_<double>(3, 3) <<
        scale, 0, -scale * centroid.x,
        0, scale, -scale * centroid.y,
        0, 0, 1);

    normPts.clear();
    for (const auto& pt : pts) {
        cv::Mat p = (cv::Mat_<double>(3, 1) << pt.x, pt.y, 1);
        cv::Mat pNorm = T * p;
        normPts.emplace_back(pNorm.at<double>(0) / pNorm.at<double>(2),
            pNorm.at<double>(1) / pNorm.at<double>(2));
    }
}


cv::Mat estimateFundamentalMatrix(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2) {
    CV_Assert(pts1.size() == pts2.size() && pts1.size() >= 8);

    std::vector<cv::Point2f> norm1, norm2;
    cv::Mat T1, T2;
    normalizePoints(pts1, norm1, T1);
    normalizePoints(pts2, norm2, T2);

    cv::Mat A(pts1.size(), 9, CV_64F);
    for (size_t i = 0; i < pts1.size(); ++i) {
        double x1 = norm1[i].x, y1 = norm1[i].y;
        double x2 = norm2[i].x, y2 = norm2[i].y;

        A.at<double>(i, 0) = x2 * x1;
        A.at<double>(i, 1) = x2 * y1;
        A.at<double>(i, 2) = x2;
        A.at<double>(i, 3) = y2 * x1;
        A.at<double>(i, 4) = y2 * y1;
        A.at<double>(i, 5) = y2;
        A.at<double>(i, 6) = x1;
        A.at<double>(i, 7) = y1;
        A.at<double>(i, 8) = 1.0;
    }

    cv::Mat u, s, vt;
    cv::SVD::compute(A, s, u, vt);
    cv::Mat F = vt.row(vt.rows - 1).reshape(0, 3);


    // Enforce rank 2
    cv::SVD::compute(F, s, u, vt);
    s.at<double>(2) = 0;
    F = u * cv::Mat::diag(s) * vt;

    // Denormalize
    F = T2.t() * F * T1;

    return F;
}

cv::Mat recoverPoseCustom(const cv::Mat& E,
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& K,
    cv::Mat& R,
    cv::Mat& t) {
    cv::Mat U, S, Vt;
    cv::SVD::compute(E, S, U, Vt);

    if (cv::determinant(U) < 0) U *= -1;
    if (cv::determinant(Vt) < 0) Vt *= -1;

    cv::Mat W = (cv::Mat_<double>(3, 3) <<
        0, -1, 0,
        1, 0, 0,
        0, 0, 1);

    cv::Mat R1 = U * W * Vt;
    cv::Mat R2 = U * W.t() * Vt;
    cv::Mat t1 = U.col(2);
    cv::Mat t2 = -U.col(2);

    std::vector<cv::Mat> R_vec{ R1, R1, R2, R2 };
    std::vector<cv::Mat> t_vec{ t1, t2, t1, t2 };

    int max_positive = -1;
    for (int i = 0; i < 4; ++i) {
        cv::Mat Rt1 = cv::Mat::eye(3, 4, CV_64F);
        cv::Mat Rt2(3, 4, CV_64F);
        R_vec[i].copyTo(Rt2(cv::Rect(0, 0, 3, 3)));
        t_vec[i].copyTo(Rt2(cv::Rect(3, 0, 1, 3)));

        cv::Mat P1 = K * Rt1;
        cv::Mat P2 = K * Rt2;

        cv::Mat pts4D;
        cv::triangulatePoints(P1, P2, pts1, pts2, pts4D);

        int count_positive = 0;
        for (int c = 0; c < pts4D.cols; ++c) {
            cv::Mat pt = pts4D.col(c);
            pt /= pt.at<float>(3);
            //if (pt.at<float>(2) > 0) count_positive++;
            cv::Mat pt4D = pts4D.col(c); // Homogeneous [X Y Z W]
            pt4D /= pt4D.at<float>(3);   // Normalize to [x y z 1]
            cv::Mat pt3D = pt4D.rowRange(0, 3); // Take [x y z]

            float z1 = pt3D.at<float>(2); // depth in first camera (should be > 0)

            // project to second camera: R * pt + t
            //cv::Mat pt2 = R_vec[i] * pt3D + t_vec[i];
            cv::Mat pt3D_double;
            pt3D.convertTo(pt3D_double, CV_64F);
            cv::Mat pt2 = R_vec[i] * pt3D_double + t_vec[i];

            float z2 = pt2.at<float>(2); // depth in second camera

            if (z1 > 0 && z2 > 0)
                count_positive++;
        }

        if (count_positive > max_positive) {
            max_positive = count_positive;
            R = R_vec[i];
            t = t_vec[i];
        }
    }

    return cv::Mat(); 
}

