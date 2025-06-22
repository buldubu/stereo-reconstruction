#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

const bool USE_GRAYSCALE = false;

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

    // Hard-coded calibration for Motorcycle-perfect
    cv::Mat K1 = (cv::Mat_<double>(3,3) << 3979.911, 0, 1244.772,
                                           0, 3979.911, 1019.507,
                                           0, 0, 1);
    cv::Mat K2 = (cv::Mat_<double>(3,3) << 3979.911, 0, 1369.115,
                                           0, 3979.911, 1019.507,
                                           0, 0, 1);
    const double baseline = 193.001;

    // Extrinsic parameters
    // R: relative rotation (identity here since camera already level)
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    // T: translation (baseline along X)
    cv::Mat T = (cv::Mat_<double>(3, 1) << -baseline, 0, 0);

    // Compute rectification matrices (R1, R2) and new projections (P1, P2, Q)
    cv::Mat R1, R2, P1, P2, Q;
    cv::Size imgSize = imgL.size();
    cv::stereoRectify(K1, cv::Mat(), K2, cv::Mat(), imgSize, R, T,
                      R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0, imgSize);

    // initUndistortRectifyMap builds (x,y) maps that warp each pixel to its rectified spot.
    cv::Mat map1x, map1y, map2x, map2y;
    cv::initUndistortRectifyMap(K1, cv::Mat(), R1, P1, imgSize, CV_32FC1, map1x, map1y);
    cv::initUndistortRectifyMap(K2, cv::Mat(), R2, P2, imgSize, CV_32FC1, map2x, map2y);

    // Apply remapping to obtain rectified images
    cv::Mat imgLRect, imgRRect;
    cv::remap(imgL, imgLRect, map1x, map1y, cv::INTER_LINEAR);
    cv::remap(imgR, imgRRect, map2x, map2y, cv::INTER_LINEAR);

    // Save rectified images
    cv::imwrite(outputName + "rectified_left.jpg", imgLRect);
    cv::imwrite(outputName + "rectified_right.jpg", imgRRect);

    // detect key-points & descriptors
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(); 
    std::vector<cv::KeyPoint> kptsL, kptsR;
    cv::Mat descL, descR;
    sift->detectAndCompute(imgLRect,  cv::noArray(), kptsL, descL);
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

    // match descriptors with Brute-Force(BF)-Matcher
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, false);
    std::vector< std::vector<cv::DMatch> > knnMatches;
    matcher->knnMatch(descL, descR, knnMatches, 2);

    const float ratioThresh = 0.2f;
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
    
    return 0;
}
