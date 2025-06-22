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
    cv::Mat imgL = cv::imread("../Data/Motorcycle-imperfect/im0.png", cv::IMREAD_COLOR);
    cv::Mat imgR = cv::imread("../Data/Motorcycle-imperfect/im1.png", cv::IMREAD_COLOR);
    
    if (USE_GRAYSCALE) {
        cv::cvtColor(imgL, imgL, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgR, imgR, cv::COLOR_BGR2GRAY);
    }

    // hard-coded from the calib file
    cv::Mat K1 = (cv::Mat_<double>(3,3) << 3997.684, 0, 1176.728,
                                       0, 3997.684, 1011.728,
                                       0, 0, 1);
    cv::Mat K2 = (cv::Mat_<double>(3,3) << 3997.684, 0, 1307.839,
                                       0, 3997.684, 1011.728,
                                       0, 0, 1);
    const double baseline = 193.001;

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

    // essential matrix and pose recovery with known intrinsics
    cv::Mat E = K2.t() * F * K1;
    cv::Mat R, T;
    //   recover the actual rotation and translation
    cv::recoverPose(E, ptsL, ptsR, K1, R, T, maskF);

    //  assume zero distortion
    cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat R1, R2, P1, P2, Q;
    cv::Size imgSize = imgL.size();
    cv::stereoRectify(K1, distCoeffs, K2, distCoeffs, imgSize, R, T,
                      R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0, imgSize);

    // build rectification maps
    cv::Mat map1x, map1y, map2x, map2y;
    cv::initUndistortRectifyMap(K1, distCoeffs, R1, P1, imgSize, CV_32FC1, map1x, map1y);
    cv::initUndistortRectifyMap(K2, distCoeffs, R2, P2, imgSize, CV_32FC1, map2x, map2y);

    // remap the pixels
    cv::Mat imgLRect, imgRRect;
    cv::remap(imgL, imgLRect, map1x, map1y, cv::INTER_LINEAR);
    cv::remap(imgR, imgRRect, map2x, map2y, cv::INTER_LINEAR);

    cv::imwrite(outputName + "rectified_left.jpg", imgLRect);
    cv::imwrite(outputName + "rectified_right.jpg", imgRRect);

    // detect key-points & descriptors, basically as before but on the rectified images
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
    
    return 0;
}
