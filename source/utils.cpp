#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <numeric>

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

// MODIFIED: Enhanced robust disparity filtering integrated into existing function
cv::Mat robustDisparityFiltering(const cv::Mat& disparity) {
    cv::Mat filtered;
    disparity.convertTo(filtered, CV_32F, 1.0 / 16.0);

    // Step 1: Remove very small disparities (cause extreme depth)
    for (int y = 0; y < filtered.rows; y++) {
        for (int x = 0; x < filtered.cols; x++) {
            float d = filtered.at<float>(y, x);
            if (d > 0 && d < 3.0f) {  // Remove disparities < 3 pixels
                filtered.at<float>(y, x) = 0;
            }
        }
    }

    // Step 2: Local consistency check - remove isolated outliers
    cv::Mat consistent = filtered.clone();
    for (int y = 2; y < filtered.rows - 2; y++) {
        for (int x = 2; x < filtered.cols - 2; x++) {
            float center = filtered.at<float>(y, x);
            if (center <= 0) continue;

            // Check 5x5 neighborhood
            std::vector<float> neighbors;
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    if (dy == 0 && dx == 0) continue;
                    float neighbor = filtered.at<float>(y + dy, x + dx);
                    if (neighbor > 0) neighbors.push_back(neighbor);
                }
            }

            if (neighbors.size() >= 5) {
                std::sort(neighbors.begin(), neighbors.end());
                float median = neighbors[neighbors.size() / 2];

                // If disparity is too different from local median, remove it
                if (std::abs(center - median) > std::max(5.0f, median * 0.3f)) {
                    consistent.at<float>(y, x) = 0;
                }
            }
            else {
                // Not enough valid neighbors - likely isolated point
                consistent.at<float>(y, x) = 0;
            }
        }
    }

    // Step 3: Apply bilateral filter for edge-preserving smoothing
    cv::Mat bilateralFiltered;
    cv::bilateralFilter(consistent, bilateralFiltered, 5, 50, 50);

    // Step 4: Light median filter
    cv::Mat result;
    cv::medianBlur(bilateralFiltered, result, 3);

    return result;
}

// robust depth outlier removal
cv::Mat removeDepthOutliersRobust(const cv::Mat& depthMap, float maxDepth = 8000.0f, float minDepth = 500.0f) {
    cv::Mat cleaned = depthMap.clone();

    // First pass - basic range filtering
    for (int y = 0; y < depthMap.rows; ++y) {
        for (int x = 0; x < depthMap.cols; ++x) {
            cv::Vec3f point = depthMap.at<cv::Vec3f>(y, x);
            float depth = point[2];

            if (depth > maxDepth || depth < minDepth || std::isnan(depth) || std::isinf(depth)) {
                cleaned.at<cv::Vec3f>(y, x) = cv::Vec3f(0, 0, 0);
            }
        }
    }

    // Second pass - local outlier detection
    cv::Mat result = cleaned.clone();
    for (int y = 2; y < depthMap.rows - 2; y++) {
        for (int x = 2; x < depthMap.cols - 2; x++) {
            cv::Vec3f center = cleaned.at<cv::Vec3f>(y, x);
            if (center[2] <= 0) continue;

            // Collect valid neighbor depths
            std::vector<float> neighborDepths;
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    if (dy == 0 && dx == 0) continue;
                    cv::Vec3f neighbor = cleaned.at<cv::Vec3f>(y + dy, x + dx);
                    if (neighbor[2] > 0) neighborDepths.push_back(neighbor[2]);
                }
            }

            if (neighborDepths.size() >= 8) {
                std::sort(neighborDepths.begin(), neighborDepths.end());
                float median = neighborDepths[neighborDepths.size() / 2];

                // More aggressive outlier detection
                float threshold = std::max(200.0f, median * 0.2f);
                if (std::abs(center[2] - median) > threshold) {
                    result.at<cv::Vec3f>(y, x) = cv::Vec3f(0, 0, 0);
                }
            }
        }
    }

    return result;
}

void writePointCloudPLY(const cv::Mat& disparityMap, const cv::Mat& depthMap, const cv::Mat& leftImage, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    int count = 0;
    // MODIFIED: Enhanced validation for point counting
    for (int y = 0; y < disparityMap.rows; ++y) {
        for (int x = 0; x < disparityMap.cols; ++x) {
            float d = disparityMap.at<float>(y, x);
            cv::Vec3f point = depthMap.at<cv::Vec3f>(y, x);
            // Better validation: check both disparity and 3D point validity
            if (d > 0.0f && point[2] > 0 && !std::isnan(point[2]) && !std::isinf(point[2]))
                count++;
        }
    }

    out << "ply\nformat ascii 1.0\n";
    out << "element vertex " << count << "\n";
    out << "property float x\nproperty float y\nproperty float z\n";
    out << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";

    for (int y = 0; y < disparityMap.rows; ++y) {
        for (int x = 0; x < disparityMap.cols; ++x) {
            float d = disparityMap.at<float>(y, x);
            if (d <= 0.0f) continue;
            cv::Vec3f pt = depthMap.at<cv::Vec3f>(y, x);
            // MODIFIED: Enhanced validation with NaN/Inf checks
            if (!cv::checkRange(pt) || pt[2] <= 0 || std::isnan(pt[2]) || std::isinf(pt[2])) continue;
            cv::Vec3b color = leftImage.at<cv::Vec3b>(y, x);
            out << pt[0] << " " << pt[1] << " " << pt[2] << " "
                << (int)color[2] << " " << (int)color[1] << " " << (int)color[0] << "\n";
        }
    }
    out.close();
    std::cout << "Point cloud saved. " << filename << std::endl;
}

// COMPLETELY REWRITTEN: Robust mesh generation with strict validation
void writeMeshFromDepth(const cv::Mat& dispFloat,
    const cv::Mat& depthMap,
    const std::string& outputObjPath,
    float max_z_diff = 150.0f)  // MODIFIED: More conservative default threshold
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

    // MODIFIED: Enhanced vertex validation with minimum disparity check
    float minDisparity = 3.0f;  // NEW: Minimum disparity threshold
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float d = dispFloat.at<float>(y, x);
            // MODIFIED: Added minimum disparity check
            if (d < minDisparity) continue;

            cv::Vec3f point = depthMap.at<cv::Vec3f>(y, x);
            // MODIFIED: Enhanced validation with NaN/Inf checks
            if (std::isnan(point[2]) || std::isinf(point[2]) || point[2] <= 0) continue;

            objFile << "v " << point[0] << " " << point[1] << " " << point[2] << "\n";
            valid_map[y * width + x] = vertex_idx++;
        }
    }

    // NEW: Added vertex count reporting
    std::cout << "Created " << (vertex_idx - 1) << " vertices with robust filtering" << std::endl;

    // MODIFIED: Enhanced triangle generation with multiple validation checks
    float maxDistanceThresh = 300.0f;  // NEW: Maximum edge length threshold
    int facesGenerated = 0;  // NEW: Face counter

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

            // MODIFIED: First triangle with enhanced validation
            if (vi0 > 0 && vi1 > 0 && vi2 > 0) {
                cv::Vec3f p0 = depthMap.at<cv::Vec3f>(y, x);
                cv::Vec3f p1 = depthMap.at<cv::Vec3f>(y, x + 1);
                cv::Vec3f p2 = depthMap.at<cv::Vec3f>(y + 1, x);

                // NEW: Multiple validation checks
                float maxZ = std::max({ p0[2], p1[2], p2[2] });
                float minZ = std::min({ p0[2], p1[2], p2[2] });
                float d01 = cv::norm(p0 - p1);
                float d02 = cv::norm(p0 - p2);
                float d12 = cv::norm(p1 - p2);

                // NEW: Enhanced triangle validation
                if ((maxZ - minZ) < max_z_diff &&
                    d01 < maxDistanceThresh && d02 < maxDistanceThresh && d12 < maxDistanceThresh &&
                    d01 > 1.0f && d02 > 1.0f && d12 > 1.0f) {  // Prevent degenerate triangles

                    // NEW: Triangle area validation
                    cv::Vec3f v1 = p1 - p0;
                    cv::Vec3f v2 = p2 - p0;
                    float area = cv::norm(v1.cross(v2)) * 0.5f;

                    if (area > 0.1f && area < 10000.0f) {  // Reasonable area range
                        objFile << "f " << vi0 << " " << vi2 << " " << vi1 << "\n";
                        facesGenerated++;
                    }
                }
            }

            // MODIFIED: Second triangle with same enhanced validation
            if (vi2 > 0 && vi1 > 0 && vi3 > 0) {
                cv::Vec3f p1 = depthMap.at<cv::Vec3f>(y, x + 1);
                cv::Vec3f p2 = depthMap.at<cv::Vec3f>(y + 1, x);
                cv::Vec3f p3 = depthMap.at<cv::Vec3f>(y + 1, x + 1);

                float maxZ = std::max({ p1[2], p2[2], p3[2] });
                float minZ = std::min({ p1[2], p2[2], p3[2] });
                float d12 = cv::norm(p1 - p2);
                float d13 = cv::norm(p1 - p3);
                float d23 = cv::norm(p2 - p3);

                if ((maxZ - minZ) < max_z_diff &&
                    d12 < maxDistanceThresh && d13 < maxDistanceThresh && d23 < maxDistanceThresh &&
                    d12 > 1.0f && d13 > 1.0f && d23 > 1.0f) {

                    cv::Vec3f v1 = p3 - p1;
                    cv::Vec3f v2 = p2 - p1;
                    float area = cv::norm(v1.cross(v2)) * 0.5f;

                    if (area > 0.1f && area < 10000.0f) {
                        objFile << "f " << vi2 << " " << vi3 << " " << vi1 << "\n";
                        facesGenerated++;
                    }
                }
            }
        }
    }
    objFile.close();
    // MODIFIED: Enhanced completion message
    std::cout << "Generated " << facesGenerated << " robust triangles" << std::endl;
    std::cout << "Mesh saved." << std::endl;
}

// MODIFIED: Enhanced SGBM creation with robust parameters
cv::Ptr<cv::StereoSGBM> createRobustSGBM(const cv::Mat& leftImg, int numDisparities = 128) {
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create();

    int blockSize = 5;
    int channels = leftImg.channels();

    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numDisparities);
    sgbm->setBlockSize(blockSize);

    // MODIFIED: More aggressive smoothing to prevent outliers
    sgbm->setP1(8 * channels * blockSize * blockSize);
    sgbm->setP2(48 * channels * blockSize * blockSize);  // Higher P2 for more smoothing

    // MODIFIED: Stricter quality controls
    sgbm->setDisp12MaxDiff(1);          // Very strict left-right check
    sgbm->setPreFilterCap(31);          // Lower pre-filter cap
    sgbm->setUniquenessRatio(15);       // Higher uniqueness requirement
    sgbm->setSpeckleWindowSize(150);    // Larger speckle removal
    sgbm->setSpeckleRange(16);          // Smaller speckle range

    sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

    return sgbm;
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
