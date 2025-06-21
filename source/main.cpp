#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>

int main(int argc, const char* argv[])
{
    std::string outputName = "../outputs/rgb_";
    // Load data
    // cv::Mat imgL = cv::imread("../Data/Motorcycle-perfect/im0.png", cv::IMREAD_GRAYSCALE);
    // cv::Mat imgR = cv::imread("../Data/Motorcycle-perfect/im1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat imgL = cv::imread("../Data/Motorcycle-perfect/im0.png", cv::IMREAD_COLOR);
    cv::Mat imgR = cv::imread("../Data/Motorcycle-perfect/im1.png", cv::IMREAD_COLOR);
    
    // convert it to grayscale for sift feature extraction
    //cv::cvtColor(imgL, grayImgL, cv::COLOR_BGR2GRAY); 
    //cv::cvtColor(imgR, grayImgR, cv::COLOR_BGR2GRAY);

    // detect key-points & descriptors
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(); 
    std::vector<cv::KeyPoint> kptsL, kptsR;
    cv::Mat descL, descR;
    sift->detectAndCompute(imgL, cv::noArray(), kptsL, descL);
    sift->detectAndCompute(imgR, cv::noArray(), kptsR, descR);

    // draw SIFT keypoints on the images
    cv::Mat imgKeypointsL, imgKeypointsR;
    cv::drawKeypoints(imgL, kptsL, imgKeypointsL, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(imgR, kptsR, imgKeypointsR, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

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
    cv::drawMatches(imgL, kptsL, imgR, kptsR,
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
