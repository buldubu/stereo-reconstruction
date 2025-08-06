#pragma once
#include <opencv2/core.hpp>

void vectorizedReprojectTo3D(const cv::Mat& disp, cv::Mat& depthMap, const cv::Mat& Q) {

    int rows = disp.rows;
    int cols = disp.cols;

    cv::Mat xMat(rows, cols, CV_32F);
    cv::Mat yMat(rows, cols, CV_32F);
    for (int y = 0; y < rows; ++y) {
        float* xRow = xMat.ptr<float>(y);
        float* yRow = yMat.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            xRow[x] = static_cast<float>(x);
            yRow[x] = static_cast<float>(y);
        }
    }

    cv::Mat ones = cv::Mat::ones(rows, cols, CV_32F);
    std::vector<cv::Mat> vecs = { xMat, yMat, disp, ones };
    cv::Mat vec4;
    cv::merge(vecs, vec4);

    cv::Mat vec4_reshaped = vec4.reshape(1, rows * cols);
    cv::Mat vec4d;
    vec4_reshaped.convertTo(vec4d, CV_64F);

    cv::Mat X = vec4d * Q.t();

    cv::Mat Xw = X.col(0) / X.col(3);
    cv::Mat Yw = X.col(1) / X.col(3);
    cv::Mat Zw = X.col(2) / X.col(3);

    cv::Mat Xwf, Ywf, Zwf;
    Xw.convertTo(Xwf, CV_32F);
    Yw.convertTo(Ywf, CV_32F);
    Zw.convertTo(Zwf, CV_32F);

    std::vector<cv::Mat> XYZ = { Xwf, Ywf, Zwf };
    cv::Mat merged = cv::Mat(Xwf.rows, 3, CV_32F);
    cv::merge(XYZ, merged);

    depthMap = merged.reshape(3, rows);

    // cv::Mat validMask = (disp > 0.0f) & (disp == disp);
    // for (int y = 0; y < rows; ++y) {
    //     for (int x = 0; x < cols; ++x) {
    //         if (!validMask.at<uchar>(y, x)) {
    //             depthMap.at<cv::Vec3f>(y, x) = cv::Vec3f(0, 0, 0);
    //         }
    //     }
    // }
}

void manualReprojectTo3D(const cv::Mat& disp, cv::Mat& depthMap, const cv::Mat& Q, const bool vectorized = false) {
    // std::cout << "Disp map size: " << disp.size() << std::endl;
    // std::cout << "Q matrix: " << Q << std::endl;
    // std::cout << "Q matrix size: " << Q.size() << std::endl;
    if (vectorized) {
        vectorizedReprojectTo3D(disp, depthMap, Q);
        return;
    }
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

            cv::Mat X = Q * vec;

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
    std::cout << "Point cloud saved. " << filename << std::endl;
}

void writeMeshFromDepth(const cv::Mat& dispFloat,
    const cv::Mat& depthMap,
    const std::string& outputObjPath,
    float max_z_diff = 5.0f)
{
    std::ofstream objFile(outputObjPath);
    if (!objFile.is_open()) {
        std::cerr << "Error opening file for mesh: " << outputObjPath << std::endl;
        return;
    }

    int height = depthMap.rows;
    int width = depthMap.cols;
    std::vector<int> valid_map(height * width, -1);
    int vertex_idx = 1;

    // Write vertices
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

    // Write faces
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
                    std::abs(z1 - z2) < max_z_diff)
                {
                    objFile << "f " << vi0 << " " << vi2 << " " << vi1 << "\n";
                }
            }

            if (vi2 > 0 && vi1 > 0 && vi3 > 0) {
                float z2 = depthMap.at<cv::Vec3f>(y + 1, x)[2];
                float z1 = depthMap.at<cv::Vec3f>(y, x + 1)[2];
                float z3 = depthMap.at<cv::Vec3f>(y + 1, x + 1)[2];

                if (std::abs(z2 - z1) < max_z_diff &&
                    std::abs(z2 - z3) < max_z_diff &&
                    std::abs(z1 - z3) < max_z_diff)
                {
                    objFile << "f " << vi2 << " " << vi3 << " " << vi1 << "\n";
                }
            }
        }
    }
    objFile.close();
    std::cout << "Mesh saved." << std::endl;
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
