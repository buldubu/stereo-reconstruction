#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <set>
#include "SemiGlobalMatching.h" 
#include "reconstruction.h"
#include "utils.cpp"
#include <filesystem>

const float scale = 0.5f;
const bool USE_GRAYSCALE = false;

struct StereoCalibrationParams {
    double fx = 3979.911 * scale;     
    double fy = 3979.911 * scale;
    double cxL = 1244.772 * scale;
    double cxR = 1369.115 * scale;
    double cy = 1019.507 * scale;

    cv::Mat distCoeffs1 = cv::Mat::zeros(5, 1, CV_64F);
    cv::Mat distCoeffs2 = cv::Mat::zeros(5, 1, CV_64F);
};


int main(int argc, const char* argv[])
{

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [sift | brisk]" << std::endl;
        return 1;
    }

    // read mode
    std::string mode = argv[1];
    std::transform(mode.begin(), mode.end(), mode.begin(),[](unsigned char c){ return std::tolower(c); });

    std::string outputPath = "../outputs/custom/" + mode + "/";
    if (!std::filesystem::exists(outputPath))
        std::filesystem::create_directories(outputPath);
    std::string outputName = outputPath + (USE_GRAYSCALE ? "grayscale_" : "rgb_");

    // Load data
    cv::Mat imgL = cv::imread("../Data/Motorcycle-perfect/im0.png", cv::IMREAD_COLOR);
    cv::Mat imgR = cv::imread("../Data/Motorcycle-perfect/im1.png", cv::IMREAD_COLOR);


    if (imgL.empty() || imgR.empty()) {
        std::cerr << "Cannot upload images!" << std::endl;
        return -1;
    }

    // scale images
    cv::resize(imgL, imgL, cv::Size(), scale, scale);
    cv::resize(imgR, imgR, cv::Size(), scale, scale);

    StereoCalibrationParams calib;

    double fx = calib.fx;
    double fy = calib.fy;
    double cxL = calib.cxL;
    double cxR = calib.cxR;
    double cy = calib.cy;


    // Q matrix based on calib.txt
    cv::Mat Q;

    // hard-coded from the calib file
    cv::Mat K1 = (cv::Mat_<double>(3, 3) << fx, 0, cxL,
        0, fx, cy,
        0, 0, 1);
    cv::Mat K2 = (cv::Mat_<double>(3, 3) << fx, 0, cxR,
        0, fx, cy,
        0, 0, 1);

    float ratioThresh;

    // detect & match features on the original images
    cv::Ptr<cv::Feature2D> feature;
    int normType;

    if(mode == "sift"){
        feature = cv::SIFT::create(); 
        normType = cv::NORM_L2;
        ratioThresh = 0.25f;
    }    
    else if(mode == "brisk"){
        feature = cv::BRISK::create(10, 2, 1.2f);
        normType = cv::NORM_HAMMING;
        ratioThresh = 0.2f;
    }

    std::vector<cv::KeyPoint> kptsL_raw, kptsR_raw;
    cv::Mat descL_raw, descR_raw;
    feature->detectAndCompute(imgL, cv::noArray(), kptsL_raw, descL_raw);
    feature->detectAndCompute(imgR, cv::noArray(), kptsR_raw, descR_raw);

    // match descriptors with Brute-Force(BF)-Matcher
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(normType, false);
    std::vector<std::vector<cv::DMatch>> knnMatches_raw;
    matcher->knnMatch(descL_raw, descR_raw, knnMatches_raw, 2);

    
    const float maxDescriptorDist = 70.0f;

    std::vector<cv::DMatch> goodMatches_raw;
    for (const auto& m : knnMatches_raw) {
        if (m.size() < 2) continue;
        if (m[0].distance < ratioThresh * m[1].distance && m[0].distance < maxDescriptorDist) {
            goodMatches_raw.push_back(m[0]);
        }
    }

    std::vector<cv::Point2f> ptsL, ptsR;
    for (const auto& m : goodMatches_raw) {
        ptsL.push_back(kptsL_raw[m.queryIdx].pt);
        ptsR.push_back(kptsR_raw[m.trainIdx].pt);
    }

    cv::Mat F;
    std::vector<cv::Point2f> inliersL, inliersR;

    //F = estimateFundamentalMatrix(inliersL, inliersR);
    std::vector<uchar> inlierMask;
    F = estimateFundamentalMatrixRANSAC(ptsL, ptsR, inlierMask);
    inliersL.clear();
    inliersR.clear();
    for (int i = 0; i < ptsL.size(); ++i) {
        if (!inlierMask[i]) continue;

        const cv::Point2f& pt1 = ptsL[i];
        const cv::Point2f& pt2 = ptsR[i];

        cv::Mat x1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1.0);
        cv::Mat l2 = F * x1;  // epipolar line in right image: ax + by + c = 0

        double a = l2.at<double>(0);
        double b = l2.at<double>(1);
        double c = l2.at<double>(2);

        double dist = std::abs(a * pt2.x + b * pt2.y + c) / std::sqrt(a * a + b * b);

        if (dist < 1.5) { // you can tune this value
            inliersL.push_back(pt1);
            inliersR.push_back(pt2);
        }
    }


    cv::Mat E = K2.t() * F * K1;
    cv::Mat R, T;
    recoverPoseCustom(E, inliersL, inliersR, K1, R, T);


    cv::Mat distCoeffs1 = cv::Mat::zeros(5, 1, CV_64F);
    cv::Mat distCoeffs2 = cv::Mat::zeros(5, 1, CV_64F);
    cv::Mat R1, R2, P1, P2;
    cv::Size imageSize = imgL.size();


    computeStereoRectification(
        K1, distCoeffs1,
        K2, distCoeffs2,
        imageSize, R, T,
        R1, R2, P1, P2, Q);


    cv::Mat imgLRect, imgRRect;
    cv::Mat map1x, map1y, map2x, map2y;


    computeRectificationMap(K1, R1, P1, imageSize, map1x, map1y);
    computeRectificationMap(K2, R2, P2, imageSize, map2x, map2y);


    std::cout << "map1x.at<float>(0,0): " << map1x.at<float>(0, 0) << std::endl;
    std::cout << "map1y.at<float>(0,0): " << map1y.at<float>(0, 0) << std::endl;


    remapBilinear(imgL, imgLRect, map1x, map1y);
    remapBilinear(imgR, imgRRect, map2x, map2y);


    cv::imwrite(outputName + mode + " sgm_rectified_left.jpg", imgLRect);
    cv::imwrite(outputName + mode + " sgm_rectified_right.jpg", imgRRect);

    // detect key-points & descriptors, basically as before but on the rectified images
    std::vector<cv::KeyPoint> kptsL, kptsR;
    cv::Mat descL, descR;
    feature->detectAndCompute(imgLRect, cv::noArray(), kptsL, descL);
    feature->detectAndCompute(imgRRect, cv::noArray(), kptsR, descR);

    // draw SIFT keypoints on the images
    cv::Mat imgKeypointsL, imgKeypointsR;
    cv::drawKeypoints(imgLRect, kptsL, imgKeypointsL, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(imgRRect, kptsR, imgKeypointsR, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // record sift results
    cv::imwrite(outputName + mode + " sgm_keypoints_left.jpg", imgKeypointsL);
    cv::imwrite(outputName + mode + " sgm_keypoints_right.jpg", imgKeypointsR);

    // match descriptors with Brute-Force(BF)-Matcher, again just as before but on rectified
    std::vector< std::vector<cv::DMatch> > knnMatches;
    matcher->knnMatch(descL, descR, knnMatches, 2);

    std::vector<cv::DMatch> goodMatches;
    for (const auto& m : knnMatches)
        if (m[0].distance < ratioThresh * m[1].distance)
            goodMatches.push_back(m[0]);

    // draw matches
    cv::Mat imgMatches;
    cv::drawMatches(imgLRect, kptsL, imgRRect, kptsR,
        goodMatches, imgMatches,
        cv::Scalar::all(-1), cv::Scalar::all(-1),
        std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // save keypoints matches
    cv::imwrite(outputName + mode + " sgm_good_matches.jpg", imgMatches);

    // Disparity map calculation
    cv::Mat imgLGray, imgRGray;
    if (USE_GRAYSCALE) {
        imgLGray = imgLRect;
        imgRGray = imgRRect;
    }
    else {
        cv::cvtColor(imgLRect, imgLGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgRRect, imgRGray, cv::COLOR_BGR2GRAY);
    }


    SemiGlobalMatching sgm(imgLGray, imgRGray, 128, 5, 15, 150, 10);  // lower L-R threshold
    sgm.SGM_process();
    cv::Mat dispFloat = sgm.get_disparity_float();

    cv::Mat disp16;
    dispFloat.convertTo(disp16, CV_16S, 16.0);
    int newSpeckleSize = 50;
    int newMaxDiff = 32;
    cv::filterSpeckles(disp16, 0, newSpeckleSize, newMaxDiff);

    disp16.convertTo(dispFloat, CV_32F, 1.0 / 16.0);

    cv::Mat dispBilateral;
    cv::bilateralFilter(dispFloat, dispBilateral, 9, 75, 75);
    dispFloat = dispBilateral.clone();

    // Mask invalid disparities
    cv::Mat dispMask = (dispFloat > 0.1f) & (dispFloat < 128.0f);
    cv::Mat dispFiltered;
    dispFloat.copyTo(dispFiltered, dispMask);

    // Scale Q matrix (due to image resizing)
    double baseline_meters = 193.001;
    cv::Mat Q_scaled = Q.clone();
    Q_scaled.at<double>(3, 2) = 1.0 / baseline_meters;
    Q_scaled.at<double>(3, 3) = -(cxL - cxR) / baseline_meters;

    // Reproject to 3D
    cv::Mat depthMap;
    manualReprojectTo3D(dispFiltered, depthMap, Q_scaled);

    // Normalize and visualize disparity
    cv::Mat dispU8;
    cv::normalize(dispFiltered, dispU8, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::Mat dispColor;
    cv::applyColorMap(dispU8, dispColor, cv::COLORMAP_JET);


    // Save results
    cv::imwrite(outputPath + mode +  " sgm_disparity_map.jpg", dispU8);
    cv::imwrite(outputPath + mode +  " sgm_disparity_map_color.jpg", dispColor);


    // Write point cloud
    writePointCloudPLY(dispFiltered, depthMap, imgL, outputPath + mode + " sgm_pointcloud.ply");

    // Write mesh
    writeColorMeshPLY(dispFloat, depthMap, imgLRect, outputPath + mode + " sgm_color_mesh.ply");

    return 0;
}
