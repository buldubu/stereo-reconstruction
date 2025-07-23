#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

const bool USE_GRAYSCALE = false;
double scale = 0.5f;

int main(int argc, const char* argv[])
{
    std::string outputName = std::string("../outputs/") + (USE_GRAYSCALE ? "grayscale_" : "rgb_");
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

    // moto
    double fx = 3979.911 * scale;
    double cxL = 1244.772 * scale;
    double cxR = 1369.115 * scale;
    double cy = 1019.507 * scale;
    double baseline = 193.001;


    // Q matrix based on calib.txt
    cv::Mat Q = (cv::Mat_<double>(4, 4) <<
        1, 0, 0, -cxL,
        0, 1, 0, -cy,
        0, 0, 0, fx,
        0, 0, 1.0 / baseline, (cxR - cxL) / baseline);   //  0, 0, -1.0 / baseline, (cxL - cxR) / baseline)

    // hard-coded from the calib file
    cv::Mat K1 = (cv::Mat_<double>(3, 3) << fx, 0, cxL,
        0, fx, cy,
        0, 0, 1);
    cv::Mat K2 = (cv::Mat_<double>(3, 3) << fx, 0, cxR,
        0, fx, cy,
        0, 0, 1);

    // detect & match features on the original images
    cv::Ptr<cv::SIFT> siftRaw = cv::SIFT::create();
    std::vector<cv::KeyPoint> kptsL_raw, kptsR_raw;
    cv::Mat descL_raw, descR_raw;
    siftRaw->detectAndCompute(imgL, cv::noArray(), kptsL_raw, descL_raw);
    siftRaw->detectAndCompute(imgR, cv::noArray(), kptsR_raw, descR_raw);

    // match descriptors with Brute-Force(BF)-Matcher
    cv::Ptr<cv::BFMatcher> matcher_raw = cv::BFMatcher::create(cv::NORM_L2, false);
    std::vector<std::vector<cv::DMatch>> knnMatches_raw;
    matcher_raw->knnMatch(descL_raw, descR_raw, knnMatches_raw, 2);

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

    /*
    // essential matrix and pose recovery with known intrinsics
    cv::Mat E = K2.t() * F * K1;
    cv::Mat R, T;
    // recover the actual rotation and translation
    cv::recoverPose(E, ptsL, ptsR, K1, R, T, maskF);

    //  assume zero distortion
    cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat R1, R2, P1, P2, Q;
    cv::Size imgSize = imgL.size();
    cv::stereoRectify(K1, distCoeffs, K2, distCoeffs, imgSize, R, T,
                      R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0, imgSize);  */

                      /*// build rectification maps
                      cv::Mat map1x, map1y, map2x, map2y;
                      cv::initUndistortRectifyMap(K1, distCoeffs, R1, P1, imgSize, CV_32FC1, map1x, map1y);
                      cv::initUndistortRectifyMap(K2, distCoeffs, R2, P2, imgSize, CV_32FC1, map2x, map2y);

                      // remap the pixels
                      cv::Mat imgLRect, imgRRect;
                      cv::remap(imgL, imgLRect, map1x, map1y, cv::INTER_LINEAR);
                      cv::remap(imgR, imgRRect, map2x, map2y, cv::INTER_LINEAR);  */

    cv::Mat imgLRect, imgRRect;

    imgLRect = imgL.clone();
    imgRRect = imgR.clone();

    cv::imwrite(outputName + "rectified_left.jpg", imgLRect);
    cv::imwrite(outputName + "rectified_right.jpg", imgRRect);

    // detect key-points & descriptors, basically as before but on the rectified images
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> kptsL, kptsR;
    cv::Mat descL, descR;
    sift->detectAndCompute(imgLRect, cv::noArray(), kptsL, descL);
    sift->detectAndCompute(imgRRect, cv::noArray(), kptsR, descR);

    // draw SIFT keypoints on the images
    cv::Mat imgKeypointsL, imgKeypointsR;
    cv::drawKeypoints(imgLRect, kptsL, imgKeypointsL, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(imgRRect, kptsR, imgKeypointsR, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // show keypoints on the images
    // cv::imshow("Keypoints - Left Image", imgKeypointsL);
    // cv::imshow("Keypoints - Right Image", imgKeypointsR);

    // record sift results
    cv::imwrite(outputName + "keypoints_left.jpg", imgKeypointsL);
    cv::imwrite(outputName + "keypoints_right.jpg", imgKeypointsR);

    // match descriptors with Brute-Force(BF)-Matcher, again just as before but on rectified
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, false);
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
    // cv::imshow("Good SIFT Matches (Left vs Right)", imgMatches);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
    cv::imwrite(outputName + "sift_good_matches.jpg", imgMatches);

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
    // sgbm->setMinDisparity(0);
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
    cv::imwrite(outputName + "disparity_map.jpg", dispVis);

    cv::Mat dispColor;
    cv::applyColorMap(dispVis, dispColor, cv::COLORMAP_JET);
    cv::imwrite(outputName + "colored_disp.jpg", dispColor);

    cv::Mat depthMap;
    cv::reprojectImageTo3D(dispFloat, depthMap, Q);

    std::vector<cv::Mat> xyz;
    cv::split(depthMap, xyz); // xyz[2] = z (depth)

    // Write to PLY 
    std::ofstream out(outputName + "pointcloud.ply");
    out << "ply\nformat ascii 1.0\n";

    // Count valid points
    int validPoints = 0;
    for (int y = 0; y < depthMap.rows; ++y) {
        for (int x = 0; x < depthMap.cols; ++x) {
            float d = dispFloat.at<float>(y, x);
            if (d > 0)
                ++validPoints;
        }
    }

    // write header
    out << "element vertex " << validPoints << "\n";
    out << "property float x\nproperty float y\nproperty float z\n";
    out << "property uchar red\nproperty uchar green\nproperty uchar blue\n"; // uncomment for color
    out << "end_header\n";

    // write points 
    for (int y = 0; y < depthMap.rows; ++y) {
        for (int x = 0; x < depthMap.cols; ++x) {
            float d = dispFloat.at<float>(y, x);
            if (d <= 0)
                continue;

            cv::Vec3f point = depthMap.at<cv::Vec3f>(y, x);

            cv::Vec3b color = imgLRect.at<cv::Vec3b>(y, x);
            out << point[0] << " " << point[1] << " " << point[2] << " "
                << (int)color[2] << " " << (int)color[1] << " " << (int)color[0] << "\n";
        }
    }

    out.close();


    // Mesh generation
    std::ofstream objFile(outputName + "mesh.obj");
    int height = depthMap.rows;
    int width = depthMap.cols;
    std::vector<int> valid_map(height * width, -1);
    int vertex_idx = 1;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float d = dispFloat.at<float>(y, x);
            if (d <= 0)
                continue;
            cv::Vec3f point = depthMap.at<cv::Vec3f>(y, x);
            objFile << "v " << point[0] << " " << point[1] << " " << point[2] << "\n";
            valid_map[y * width + x] = vertex_idx++;
        }
    }
    float max_z_diff = 5.0f;
    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            int i0 = y * width + x;
            int i1 = y * width + (x + 1);
            int i2 = (y + 1) * width + x;
            int i3 = (y + 1) * width + (x + 1);
            int vi0 = valid_map[i0];
            int vi1 = valid_map[i1];
            int vi2 = valid_map[i2];
            int vi3 = valid_map[i3];
            if (vi0 > 0 && vi1 > 0 && vi2 > 0) {
                float z0 = depthMap.at<cv::Vec3f>(y, x)[2];
                float z1 = depthMap.at<cv::Vec3f>(y, x + 1)[2];
                float z2 = depthMap.at<cv::Vec3f>(y + 1, x)[2];
                if (std::abs(z0 - z1) < max_z_diff &&
                    std::abs(z0 - z2) < max_z_diff &&
                    std::abs(z1 - z2) < max_z_diff) {
                    objFile << "f " << vi0 << " " << vi2 << " " << vi1 << "\n";
                }
            }
            if (vi2 > 0 && vi1 > 0 && vi3 > 0) {
                float z2 = depthMap.at<cv::Vec3f>(y + 1, x)[2];
                float z1 = depthMap.at<cv::Vec3f>(y, x + 1)[2];
                float z3 = depthMap.at<cv::Vec3f>(y + 1, x + 1)[2];
                if (std::abs(z2 - z1) < max_z_diff &&
                    std::abs(z2 - z3) < max_z_diff &&
                    std::abs(z1 - z3) < max_z_diff) {
                    objFile << "f " << vi2 << " " << vi3 << " " << vi1 << "\n";
                }
            }
        }
    }
    objFile.close();

    return 0;
}