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

void writeColorMeshPLY(const cv::Mat& dispFloat,
                       const cv::Mat& depthMap,
                       const cv::Mat& colorImage,
                       const std::string& filename,
                       float max_z_diff = 5.0f)
{
    std::ofstream ply(filename);
    if (!ply.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    int height = depthMap.rows;
    int width = depthMap.cols;

    std::vector<int> valid_map(height * width, -1);
    std::vector<cv::Vec3f> vertices;
    std::vector<cv::Vec3b> colors;
    std::vector<std::array<int, 3>> faces;

    int index = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float d = dispFloat.at<float>(y, x);
            if (d <= 0) continue;

            cv::Vec3f pt = depthMap.at<cv::Vec3f>(y, x);
            if (!cv::checkRange(pt)) continue;

            vertices.push_back(pt);
            colors.push_back(colorImage.at<cv::Vec3b>(y, x));
            valid_map[y * width + x] = index++;
        }
    }

    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            int i0 = y * width + x;
            int i1 = y * width + (x + 1);
            int i2 = (y + 1) * width + x;
            int i3 = (y + 1) * width + (x + 1);

            int v0 = valid_map[i0];
            int v1 = valid_map[i1];
            int v2 = valid_map[i2];
            int v3 = valid_map[i3];

            if (v0 >= 0 && v1 >= 0 && v2 >= 0) {
                float z0 = depthMap.at<cv::Vec3f>(y, x)[2];
                float z1 = depthMap.at<cv::Vec3f>(y, x + 1)[2];
                float z2 = depthMap.at<cv::Vec3f>(y + 1, x)[2];
                if (std::abs(z0 - z1) < max_z_diff &&
                    std::abs(z0 - z2) < max_z_diff &&
                    std::abs(z1 - z2) < max_z_diff)
                {
                    faces.push_back({v0, v2, v1});
                }
            }

            if (v2 >= 0 && v1 >= 0 && v3 >= 0) {
                float z2 = depthMap.at<cv::Vec3f>(y + 1, x)[2];
                float z1 = depthMap.at<cv::Vec3f>(y, x + 1)[2];
                float z3 = depthMap.at<cv::Vec3f>(y + 1, x + 1)[2];
                if (std::abs(z2 - z1) < max_z_diff &&
                    std::abs(z2 - z3) < max_z_diff &&
                    std::abs(z1 - z3) < max_z_diff)
                {
                    faces.push_back({v2, v3, v1});
                }
            }
        }
    }

    // Write header
    ply << "ply\nformat ascii 1.0\n";
    ply << "element vertex " << vertices.size() << "\n";
    ply << "property float x\nproperty float y\nproperty float z\n";
    ply << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    ply << "element face " << faces.size() << "\n";
    ply << "property list uchar int vertex_indices\n";
    ply << "end_header\n";

    // Write vertices
    for (size_t i = 0; i < vertices.size(); ++i) {
        const auto& pt = vertices[i];
        const auto& color = colors[i];
        ply << pt[0] << " " << pt[1] << " " << pt[2] << " "
            << (int)color[2] << " " << (int)color[1] << " " << (int)color[0] << "\n";
    }

    // Write faces
    for (const auto& f : faces) {
        ply << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";
    }

    ply.close();
    std::cout << "Colorful mesh saved to " << filename << std::endl;
}

