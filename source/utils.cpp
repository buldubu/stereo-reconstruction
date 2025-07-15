#pragma once
#include <opencv2/core.hpp>

void manualReprojectTo3D(const cv::Mat& disp, cv::Mat& depthMap, const cv::Mat& Q) {
    // std::cout << "Disp map size: " << disp.size() << std::endl;
    // std::cout << "Q matrix: " << Q << std::endl;
    // std::cout << "Q matrix size: " << Q.size() << std::endl;
    depthMap = cv::Mat(disp.size(), CV_32FC3);

    for (int y = 0; y < disp.rows; ++y) {
        for (int x = 0; x < disp.cols; ++x) {
            float d = disp.at<float>(y, x);

            if (d <= 0.0f || std::isinf(d)) {
                depthMap.at<cv::Vec3f>(y, x) = cv::Vec3f(0, 0, 0);
                continue;
            }

            cv::Mat_<double> vec(4, 1);
            vec(0) = static_cast<double>(x);
            vec(1) = static_cast<double>(y);
            vec(2) = static_cast<double>(d);
            vec(3) = 1.0;

            cv::Mat X = Q * vec;  // 4x1

            float Xw = static_cast<float>(X.at<double>(0) / X.at<double>(3));
            float Yw = static_cast<float>(X.at<double>(1) / X.at<double>(3));
            float Zw = static_cast<float>(X.at<double>(2) / X.at<double>(3));

            depthMap.at<cv::Vec3f>(y, x) = cv::Vec3f(Xw, Yw, Zw);
        }
    }
}

void writePointCloudPLY(const cv::Mat& disparityMap, const cv::Mat& depthMap, const cv::Mat& leftImage, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    int count = 0;
    for (int y = 0; y < disparityMap.rows; ++y)
        for (int x = 0; x < disparityMap.cols; ++x)
            if (disparityMap.at<float>(y, x) > 0.0f)
                count++;

    out << "ply\nformat ascii 1.0\n";
    out << "element vertex " << count << "\n";
    out << "property float x\nproperty float y\nproperty float z\n";
    out << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";

    for (int y = 0; y < disparityMap.rows; ++y) {
        for (int x = 0; x < disparityMap.cols; ++x) {
            float d = disparityMap.at<float>(y, x);
            if (d <= 0.0f) continue;
            cv::Vec3f pt = depthMap.at<cv::Vec3f>(y, x);
            if (!cv::checkRange(pt)) continue;
            cv::Vec3b color = leftImage.at<cv::Vec3b>(y, x);
            out << pt[0] << " " << pt[1] << " " << pt[2] << " "
                << (int)color[2] << " " << (int)color[1] << " " << (int)color[0] << "\n";
        }
    }
    out.close();
    std::cout << "[INFO] Point cloud saved to: " << filename << std::endl;
}

cv::Mat computeDisparitySAD(const cv::Mat& imgL, const cv::Mat& imgR, int minDisparity, int maxDisparity, int blockSize)
{
    CV_Assert(imgL.size() == imgR.size());
    CV_Assert(imgL.type() == CV_8UC1 && imgR.type() == CV_8UC1);

    const int h = imgL.rows;
    const int w = imgL.cols;
    const int halfBlock = blockSize / 2;
    const int disparityRange = maxDisparity - minDisparity + 1;

    cv::Mat disparity(h, w, CV_16S, cv::Scalar(-16));

    cv::parallel_for_(cv::Range(halfBlock, h - halfBlock), [&](const cv::Range& range)
    {
        std::vector<int> costs(disparityRange);
        
        for (int y = range.start; y < range.end; ++y)
        {
            for (int x = halfBlock + maxDisparity; x < w - halfBlock; ++x)
            {
                std::fill(costs.begin(), costs.end(), std::numeric_limits<int>::max());

                for (int d = minDisparity; d <= maxDisparity; ++d)
                {
                    int sad = 0;
                    bool valid = true;

                    if (blockSize <= 7)
                    {
                        for (int i = -halfBlock; i <= halfBlock && valid; ++i)
                        {
                            const uchar* leftRow = imgL.ptr<uchar>(y + i);
                            const uchar* rightRow = imgR.ptr<uchar>(y + i);
                            
                            for (int j = -halfBlock; j <= halfBlock; ++j)
                            {
                                int leftX = x + j;
                                int rightX = leftX - d;

                                if (rightX < 0 || rightX >= w)
                                {
                                    valid = false;
                                    break;
                                }

                                sad += std::abs(leftRow[leftX] - rightRow[rightX]);
                            }
                        }
                    }
                    else
                    {
                        for (int i = -halfBlock; i <= halfBlock && valid; ++i)
                        {
                            for (int j = -halfBlock; j <= halfBlock; ++j)
                            {
                                int leftX = x + j;
                                int rightX = leftX - d;

                                if (rightX < 0 || rightX >= w)
                                {
                                    valid = false;
                                    break;
                                }

                                int leftPixel = imgL.at<uchar>(y + i, leftX);
                                int rightPixel = imgR.at<uchar>(y + i, rightX);
                                sad += std::abs(leftPixel - rightPixel);
                            }
                        }
                    }

                    if (valid)
                        costs[d - minDisparity] = sad;
                }

                int bestD = 0;
                int minSAD = std::numeric_limits<int>::max();
                for (int d = 0; d < disparityRange; ++d)
                {
                    if (costs[d] < minSAD)
                    {
                        minSAD = costs[d];
                        bestD = d;
                    }
                }

                float disp = static_cast<float>(bestD + minDisparity);
                if (bestD > 0 && bestD < disparityRange - 1 && 
                    costs[bestD - 1] != std::numeric_limits<int>::max() &&
                    costs[bestD + 1] != std::numeric_limits<int>::max())
                {
                    int c0 = costs[bestD - 1];
                    int c1 = costs[bestD];
                    int c2 = costs[bestD + 1];
                    int denom = 2 * (c0 + c2 - 2 * c1);
                    if (denom != 0)
                        disp += static_cast<float>(c0 - c2) / denom;
                }

                disparity.at<short>(y, x) = static_cast<short>(disp * 16);
            }
        }
    });

    return disparity;
} 
