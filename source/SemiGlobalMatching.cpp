#include "SemiGlobalMatching.h"
#include <iostream>
#include <bitset>
#include <climits>

SemiGlobalMatching::SemiGlobalMatching(
    const cv::Mat& left,
    const cv::Mat& right,
    int max_disp,
    int census_size,
    int filter_size,
    int occlusion_seuil,
    int penalty_1,
    int penalty_2
) : max_disp(max_disp),
    census_size(census_size),
    filter_size(filter_size),
    occlusion_seuil(occlusion_seuil),
    penalty_1(penalty_1),
    penalty_2(penalty_2)
{
    left_I = left.clone();
    right_I = right.clone();
    height = left_I.rows;
    width = left_I.cols;

    disparities = cv::Mat::zeros(height, width, CV_8U);
    disparity_f32 = cv::Mat::zeros(height, width, CV_32F);

    left_cost_volume.resize(max_disp);
    for(int d = 0; d < max_disp; ++d)
        left_cost_volume[d] = cv::Mat::zeros(height, width, CV_32S);

    aggregation_volume.resize(4);
    for(int dir = 0; dir < 4; ++dir) {
        aggregation_volume[dir].resize(max_disp);
        for(int d = 0; d < max_disp; ++d)
            aggregation_volume[dir][d] = cv::Mat::zeros(height, width, CV_32S);
    }
}

void SemiGlobalMatching::SGM_process() {
    compute_costs();
    aggregate_costs();

    cv::Mat sum_vol = cv::Mat::zeros(height, width, CV_32S);
    for(int d = 0; d < max_disp; ++d) {
        cv::Mat sum_dir = cv::Mat::zeros(height, width, CV_32S);
        for(int dir = 0; dir < 4; ++dir)
            cv::add(sum_dir, aggregation_volume[dir][d], sum_dir);

        for(int y = 0; y < height; ++y) {
            for(int x = 0; x < width; ++x) {
                int v = sum_dir.at<int>(y, x);
                if(d == 0 || v < sum_vol.at<int>(y, x)) {
                    sum_vol.at<int>(y, x) = v;
                    disparities.at<uchar>(y, x) = static_cast<uchar>(d);
                    disparity_f32.at<float>(y, x) = static_cast<float>(d);
                }
            }
        }
    }

    disparities.convertTo(disparities, CV_8U, 255.0 / max_disp);

    for(int y = 0; y < height; ++y) {
        for(int x = 0; x < width; ++x) {
            if(disparities.at<uchar>(y, x) <= occlusion_seuil) {
                disparities.at<uchar>(y, x) = 0;
                disparity_f32.at<float>(y, x) = -1.0f;
            }
        }
    }

    cv::medianBlur(disparities, disparities, filter_size);
}

cv::Mat SemiGlobalMatching::get_disparity() const {
    return disparities.clone();
}

cv::Mat SemiGlobalMatching::get_disparity_float() const {
    return disparity_f32.clone();
}

void SemiGlobalMatching::compute_costs() {
    cv::Mat lc = cv::Mat::zeros(height, width, CV_64F);
    cv::Mat rc = cv::Mat::zeros(height, width, CV_64F);
    int half = census_size / 2;
    int bits_len = census_size * census_size - 1;

    for(int y = half; y < height - half; ++y) {
        for(int x = half; x < width - half; ++x) {
            std::string bits;
            bits.reserve(bits_len);
            int cvL = left_I.at<int>(y, x);
            for(int dy = -half; dy <= half; ++dy) {
                for(int dx = -half; dx <= half; ++dx) {
                    if(dy == 0 && dx == 0) continue;
                    bits.push_back(left_I.at<int>(y + dy, x + dx) < cvL ? '1' : '0');
                }
            }
            lc.at<double>(y, x) = std::bitset<64>(bits).to_ullong();

            bits.clear();
            int cvR = right_I.at<int>(y, x);
            for(int dy = -half; dy <= half; ++dy) {
                for(int dx = -half; dx <= half; ++dx) {
                    if(dy == 0 && dx == 0) continue;
                    bits.push_back(right_I.at<int>(y + dy, x + dx) < cvR ? '1' : '0');
                }
            }
            rc.at<double>(y, x) = std::bitset<64>(bits).to_ullong();
        }
    }

    cv::Mat rcensus = cv::Mat::zeros(height, width, CV_64F);
    for(int d = 0; d < max_disp; ++d) {
        rcensus.setTo(0);
        cv::Rect src(half, 0, width - half - d, height);
        cv::Rect dst(half + d, 0, width - half - d, height);
        rc(src).copyTo(rcensus(dst));

        for(int y = half; y < height - half; ++y) {
            for(int x = half; x < width - half; ++x) {
                if(x - d < half) continue;
                uint64_t L = (uint64_t)lc.at<double>(y, x);
                uint64_t R = (uint64_t)rcensus.at<double>(y, x);
                uint64_t X = L ^ R;
                int cnt = 0;
                while(X) { X &= (X - 1); ++cnt; }
                left_cost_volume[d].at<int>(y, x) = cnt;
            }
        }
    }
}

cv::Mat SemiGlobalMatching::get_path_cost(const cv::Mat& block) {
    int N = block.rows;
    cv::Mat path = cv::Mat::zeros(N, max_disp, CV_32S);
    cv::Mat pen = cv::Mat::zeros(max_disp, max_disp, CV_32S);
    for(int i = 0; i < max_disp; ++i) {
        for(int j = 0; j < max_disp; ++j) {
            int diff = std::abs(i - j);
            if(diff == 1) pen.at<int>(i, j) = penalty_1;
            else if(diff > 1) pen.at<int>(i, j) = penalty_2;
        }
    }

    for(int d = 0; d < max_disp; ++d)
        path.at<int>(0, d) = block.at<int>(0, d);

    for(int i = 1; i < N; ++i) {
        int pm = INT_MAX;
        for(int d = 0; d < max_disp; ++d)
            pm = std::min(pm, path.at<int>(i - 1, d));

        for(int d = 0; d < max_disp; ++d) {
            int best = INT_MAX;
            for(int dp = 0; dp < max_disp; ++dp)
                best = std::min(best, path.at<int>(i - 1, dp) + pen.at<int>(dp, d));
            path.at<int>(i, d) = block.at<int>(i, d) + best - pm;
        }
    }

    return path;
}

void SemiGlobalMatching::aggregate_costs() {
    for(int x = 0; x < width; ++x) {
        cv::Mat col = cv::Mat::zeros(height, max_disp, CV_32S);
        for(int y = 0; y < height; ++y)
            for(int d = 0; d < max_disp; ++d)
                col.at<int>(y, d) = left_cost_volume[d].at<int>(y, x);

        cv::Mat north = get_path_cost(col);
        for(int y = 0; y < height; ++y)
            for(int d = 0; d < max_disp; ++d)
                aggregation_volume[0][d].at<int>(y, x) = north.at<int>(y, d);

        cv::Mat f1, f2;
        cv::flip(col, f1, 0);
        f2 = get_path_cost(f1);
        cv::flip(f2, f2, 0);
        for(int y = 0; y < height; ++y)
            for(int d = 0; d < max_disp; ++d)
                aggregation_volume[1][d].at<int>(y, x) = f2.at<int>(y, d);
    }

    for(int y = 0; y < height; ++y) {
        cv::Mat row = cv::Mat::zeros(width, max_disp, CV_32S);
        for(int x = 0; x < width; ++x)
            for(int d = 0; d < max_disp; ++d)
                row.at<int>(x, d) = left_cost_volume[d].at<int>(y, x);

        cv::Mat west = get_path_cost(row);
        for(int x = 0; x < width; ++x)
            for(int d = 0; d < max_disp; ++d)
                aggregation_volume[2][d].at<int>(y, x) = west.at<int>(x, d);

        cv::Mat f1, f2;
        cv::flip(row, f1, 0);
        f2 = get_path_cost(f1);
        cv::flip(f2, f2, 0);
        for(int x = 0; x < width; ++x)
            for(int d = 0; d < max_disp; ++d)
                aggregation_volume[3][d].at<int>(y, x) = f2.at<int>(x, d);
    }
}
