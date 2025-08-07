#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include "utils.cpp"
#include <filesystem>

const bool USE_GRAYSCALE = false;
double scale = 0.5f;

int main(int argc, const char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [sift | brisk | clip]" << std::endl;
        return 1;
    }

    // read mode
    std::string mode = argv[1];
    std::transform(mode.begin(), mode.end(), mode.begin(),[](unsigned char c){ return std::tolower(c); });

    std::string outputPath = "../outputs/opencv/" + mode + "/";
    if (!std::filesystem::exists(outputPath)) {
    std::filesystem::create_directories(outputPath);}

    std::string outputName = outputPath + (USE_GRAYSCALE ? "grayscale_" : "rgb_"); 

    // Load data
    cv::Mat imgL = cv::imread("../Data/Motorcycle-perfect/im0.png", cv::IMREAD_COLOR);
    cv::Mat imgR = cv::imread("../Data/Motorcycle-perfect/im1.png", cv::IMREAD_COLOR);

    if (USE_GRAYSCALE) {
        cv::cvtColor(imgL, imgL, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgR, imgR, cv::COLOR_BGR2GRAY);
    }

    // scale images
    cv::resize(imgL, imgL, cv::Size(), scale, scale);
    cv::resize(imgR, imgR, cv::Size(), scale, scale);

    // camera params
    double fx = 3979.911 * scale;
    double cxL = 1244.772 * scale;
    double cxR = 1369.115 * scale;
    double cy = 1019.507 * scale;
    double baseline = 193.001;

    // Q matrix calculation
    cv::Mat Q = (cv::Mat_<double>(4, 4) <<
        1, 0, 0, -cxL,
        0, 1, 0, -cy,
        0, 0, 0, fx,
        0, 0, 1.0 / baseline, (cxR - cxL) / baseline);   //  0, 0, -1.0 / baseline, (cxL - cxR) / baseline)

    cv::Mat K1 = (cv::Mat_<double>(3, 3) << fx, 0, cxL,
        0, fx, cy,
        0, 0, 1);
    cv::Mat K2 = (cv::Mat_<double>(3, 3) << fx, 0, cxR,
        0, fx, cy,
        0, 0, 1);

    // detect & match features on the original images
    cv::Ptr<cv::Feature2D> feature;
    int normType;

    if(mode == "sift"){
        feature = cv::SIFT::create(); 
        normType = cv::NORM_L2;
    }    
    else if(mode == "brisk"){
        feature = cv::BRISK::create(10, 2, 1.2f);
        normType = cv::NORM_HAMMING;
    }
    else if(mode == "clip")
          std::cout << ""; // SELEN 

    std::vector<cv::KeyPoint> kptsL_raw, kptsR_raw;
    cv::Mat descL_raw, descR_raw;
    feature->detectAndCompute(imgL, cv::noArray(), kptsL_raw, descL_raw);
    feature->detectAndCompute(imgR, cv::noArray(), kptsR_raw, descR_raw);

    // match descriptors with Brute-Force(BF)-Matcher
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(normType, false);
    std::vector<std::vector<cv::DMatch>> knnMatches_raw;
    matcher->knnMatch(descL_raw, descR_raw, knnMatches_raw, 2);

    const float ratioThresh = 0.2f;
    std::vector<cv::DMatch> goodMatches_raw;
    for (const auto& m : knnMatches_raw)
        if (m[0].distance < ratioThresh * m[1].distance)
            goodMatches_raw.push_back(m[0]);

    std::vector<cv::Point2f> ptsL, ptsR;
    for (const auto& m : goodMatches_raw) {
        ptsL.push_back(kptsL_raw[m.queryIdx].pt);
        ptsR.push_back(kptsR_raw[m.trainIdx].pt);
    }

    //  estimate the fundamental matrix with RANSAC and reject outliers
    cv::Mat maskF;
    cv::Mat F = cv::findFundamentalMat(ptsL, ptsR, cv::FM_RANSAC, 3.0, 0.99, maskF);

    cv::Mat imgLRect, imgRRect;

    imgLRect = imgL.clone();
    imgRRect = imgR.clone();

    cv::imwrite(outputName + mode + " opencv_rectified_left.jpg", imgLRect);
    cv::imwrite(outputName + mode + " opencv_rectified_right.jpg", imgRRect);

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
    cv::imwrite(outputName + mode + " opencv_keypoints_left.jpg", imgKeypointsL);
    cv::imwrite(outputName + mode + " opencv_keypoints_right.jpg", imgKeypointsR);

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
    cv::imwrite(outputName + mode + " opencv_sift_good_matches.jpg", imgMatches);

    // disparity map calculation
    cv::Mat imgLGray, imgRGray;
    if (USE_GRAYSCALE) {
        imgLGray = imgLRect;
        imgRGray = imgRRect;
    }
    else {
        cv::cvtColor(imgLRect, imgLGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgRRect, imgRGray, cv::COLOR_BGR2GRAY);
    }

    int blockSize = 5;
    int numDisparities = 128;
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, numDisparities, blockSize);

    sgbm->setBlockSize(blockSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numDisparities);
    sgbm->setP1(8 * imgLGray.channels() * blockSize * blockSize);
    sgbm->setP2(32 * imgLGray.channels() * blockSize * blockSize);
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

    cv::Mat disp;
    sgbm->compute(imgLGray, imgRGray, disp);

    cv::Mat dispFloat;
    disp.convertTo(dispFloat, CV_32F, 1.0 / 16.0);

    cv::Mat dispVis;
    cv::normalize(dispFloat, dispVis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite(outputPath + mode + " opencv_disparity_map.jpg", dispVis);

    cv::Mat dispColor;
    cv::applyColorMap(dispVis, dispColor, cv::COLORMAP_JET);
    cv::imwrite(outputPath + mode + " opencv_colored_disp.jpg", dispColor);

    cv::Mat depthMap;
    cv::reprojectImageTo3D(dispFloat, depthMap, Q);

    std::vector<cv::Mat> xyz;
    cv::split(depthMap, xyz); 

    writePointCloudPLY(dispFloat, depthMap, imgLRect, outputPath + mode + " opencv_pointcloud.ply");
    writeColorMeshPLY(dispFloat, depthMap, imgLRect, outputPath + mode + " opencv_color_mesh.ply");
    
    return 0;
}