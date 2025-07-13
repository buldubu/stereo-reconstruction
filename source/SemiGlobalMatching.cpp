#include "SemiGlobalMatching.h"
#include <iostream>
#include <bitset>
#include <climits>
#include <algorithm>

SemiGlobalMatching::SemiGlobalMatching(
    const cv::Mat& left,
    const cv::Mat& right,
    int max_disp,
    int census_size,
    int penalty_1,
    int penalty_2,
    float lr_threshold
) : max_disp(max_disp),
    census_size(census_size),
    penalty_1(penalty_1),
    penalty_2(penalty_2),
    lr_threshold(lr_threshold)
{
    if (left.channels() == 3) {
        cv::cvtColor(left, left_I, cv::COLOR_BGR2GRAY);
        cv::cvtColor(right, right_I, cv::COLOR_BGR2GRAY);
    } else {
        left_I = left.clone();
        right_I = right.clone();
    }
    
    if (left_I.type() != CV_8U) {
        left_I.convertTo(left_I, CV_8U);
        right_I.convertTo(right_I, CV_8U);
    }
    
    height = left_I.rows;
    width = left_I.cols;

    disparities = cv::Mat::zeros(height, width, CV_8U);
    disparity_f32 = cv::Mat::zeros(height, width, CV_32F);
    disparity_right = cv::Mat::zeros(height, width, CV_32F);

    left_census = cv::Mat::zeros(height, width, CV_64F);
    right_census = cv::Mat::zeros(height, width, CV_64F);

    cost_volume = cv::Mat::zeros(height * width, max_disp, CV_16U);
    aggregated_cost = cv::Mat::zeros(height * width, max_disp, CV_32S);
}

void SemiGlobalMatching::SGM_process() {
    std::cout << "Computing census transform..." << std::endl;
    compute_census_transform();
    
    std::cout << "Computing matching costs..." << std::endl;
    compute_matching_cost();
    
    std::cout << "Aggregating costs..." << std::endl;
    aggregate_costs();
    
    std::cout << "Computing disparity (WTA)..." << std::endl;
    compute_disparity_WTA();
    
    std::cout << "Computing right disparity..." << std::endl;
    compute_right_disparity();
    
    std::cout << "Left-right consistency check..." << std::endl;
    left_right_consistency_check();
    
    std::cout << "Sub-pixel refinement..." << std::endl;
    sub_pixel_refinement();
    
    std::cout << "Median filtering..." << std::endl;
    median_filter();
    
    std::cout << "SGM processing complete!" << std::endl;
}

void SemiGlobalMatching::compute_census_transform() {
    int half = census_size / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (y < half || y >= height - half || x < half || x >= width - half) {
                left_census.at<double>(y, x) = 0;
                right_census.at<double>(y, x) = 0;
                continue;
            }
            
            uint64_t left_code = compute_census_at(left_I, x, y);
            uint64_t right_code = compute_census_at(right_I, x, y);
            
            left_census.at<double>(y, x) = static_cast<double>(left_code);
            right_census.at<double>(y, x) = static_cast<double>(right_code);
        }
    }
}

uint64_t SemiGlobalMatching::compute_census_at(const cv::Mat& img, int x, int y) const {
    uint64_t census = 0;
    int bit_pos = 0;
    int half = census_size / 2;
    
    uchar center_val = img.at<uchar>(y, x);
    
    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            if (dy == 0 && dx == 0) continue;
            
            uchar neighbor_val = img.at<uchar>(y + dy, x + dx);
            if (neighbor_val < center_val) {
                census |= (1ULL << bit_pos);
            }
            bit_pos++;
        }
    }
    
    return census;
}

int SemiGlobalMatching::hamming_distance(uint64_t a, uint64_t b) const {
    uint64_t xor_result = a ^ b;
    return __builtin_popcountll(xor_result);
}

void SemiGlobalMatching::compute_matching_cost() {
    int half = census_size / 2;
    
    for (int y = half; y < height - half; ++y) {
        for (int x = half; x < width - half; ++x) {
            uint64_t left_code = static_cast<uint64_t>(left_census.at<double>(y, x));
            
            for (int d = 0; d < max_disp; ++d) {
                int right_x = x - d;
                uint16_t cost = 255; // Maximum cost for invalid pixels
                
                if (right_x >= half && right_x < width - half) {
                    uint64_t right_code = static_cast<uint64_t>(right_census.at<double>(y, right_x));
                    cost = static_cast<uint16_t>(std::min(255, hamming_distance(left_code, right_code) * 4));
                }
                
                cost_volume.at<uint16_t>(y * width + x, d) = cost;
            }
        }
    }
}

void SemiGlobalMatching::aggregate_costs() {

    aggregated_cost.setTo(0);
    
    // Aggregate costs from all 8 directions
    for (int path = 0; path < NUM_PATHS; ++path) {
        int dx = dir_x[path];
        int dy = dir_y[path];
        
        // Determine starting points for this direction
        if (dx == 0) { // Vertical paths
            if (dy < 0) { // North
                for (int x = 0; x < width; ++x) {
                    aggregate_path(x, height - 1, dx, dy);
                }
            } else { // South
                for (int x = 0; x < width; ++x) {
                    aggregate_path(x, 0, dx, dy);
                }
            }
        } else if (dy == 0) { // Horizontal paths
            if (dx < 0) { // West
                for (int y = 0; y < height; ++y) {
                    aggregate_path(width - 1, y, dx, dy);
                }
            } else { // East
                for (int y = 0; y < height; ++y) {
                    aggregate_path(0, y, dx, dy);
                }
            }
        } else { // Diagonal paths
            if (dx < 0 && dy < 0) { // Northwest
                for (int x = 0; x < width; ++x) {
                    aggregate_path(x, height - 1, dx, dy);
                }
                for (int y = 0; y < height - 1; ++y) {
                    aggregate_path(width - 1, y, dx, dy);
                }
            } else if (dx > 0 && dy < 0) { // Northeast
                for (int x = 0; x < width; ++x) {
                    aggregate_path(x, height - 1, dx, dy);
                }
                for (int y = 0; y < height - 1; ++y) {
                    aggregate_path(0, y, dx, dy);
                }
            } else if (dx < 0 && dy > 0) { // Southwest
                for (int x = 0; x < width; ++x) {
                    aggregate_path(x, 0, dx, dy);
                }
                for (int y = 1; y < height; ++y) {
                    aggregate_path(width - 1, y, dx, dy);
                }
            } else { // Southeast
                for (int x = 0; x < width; ++x) {
                    aggregate_path(x, 0, dx, dy);
                }
                for (int y = 1; y < height; ++y) {
                    aggregate_path(0, y, dx, dy);
                }
            }
        }
    }
}

void SemiGlobalMatching::aggregate_path(int start_x, int start_y, int dx, int dy) {
    std::vector<int> prev_costs(max_disp, 0);
    std::vector<int> curr_costs(max_disp, 0);
    
    int x = start_x;
    int y = start_y;
    
    if (is_valid_pixel(x, y)) {
        for (int d = 0; d < max_disp; ++d) {
            prev_costs[d] = cost_volume.at<uint16_t>(y * width + x, d);
        }
    }

    x += dx;
    y += dy;
    
    while (is_valid_pixel(x, y)) {
        int prev_min = *std::min_element(prev_costs.begin(), prev_costs.end());
        
        for (int d = 0; d < max_disp; ++d) {
            int raw_cost = cost_volume.at<uint16_t>(y * width + x, d);
            
            int min_penalty = INT_MAX;
            
            min_penalty = std::min(min_penalty, prev_costs[d]);
            
            if (d > 0) {
                min_penalty = std::min(min_penalty, prev_costs[d - 1] + penalty_1);
            }
            if (d < max_disp - 1) {
                min_penalty = std::min(min_penalty, prev_costs[d + 1] + penalty_1);
            }
            
            min_penalty = std::min(min_penalty, prev_min + penalty_2);
            
            curr_costs[d] = raw_cost + min_penalty - prev_min;
        }

        for (int d = 0; d < max_disp; ++d) {
            aggregated_cost.at<int>(y * width + x, d) += curr_costs[d];
        }
        
        prev_costs = curr_costs;

        x += dx;
        y += dy;
    }
}

void SemiGlobalMatching::compute_disparity_WTA() {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int min_cost = INT_MAX;
            int best_d = 0;
            
            for (int d = 0; d < max_disp; ++d) {
                int cost = aggregated_cost.at<int>(y * width + x, d);
                if (cost < min_cost) {
                    min_cost = cost;
                    best_d = d;
                }
            }
            
            disparities.at<uchar>(y, x) = static_cast<uchar>(best_d);
            disparity_f32.at<float>(y, x) = static_cast<float>(best_d);
        }
    }
}

void SemiGlobalMatching::compute_right_disparity() {
    // Similar to left disparity but for right image
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int min_cost = INT_MAX;
            int best_d = 0;
            
            for (int d = 0; d < max_disp; ++d) {
                int left_x = x + d;
                if (left_x >= width) continue;
                
                int cost = aggregated_cost.at<int>(y * width + left_x, d);
                if (cost < min_cost) {
                    min_cost = cost;
                    best_d = d;
                }
            }
            
            disparity_right.at<float>(y, x) = static_cast<float>(best_d);
        }
    }
}

void SemiGlobalMatching::left_right_consistency_check() {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float left_disp = disparity_f32.at<float>(y, x);
            int right_x = x - static_cast<int>(left_disp);
            
            if (right_x < 0 || right_x >= width) {
                disparity_f32.at<float>(y, x) = -1.0f;  
                disparities.at<uchar>(y, x) = 0;
                continue;
            }
            
            float right_disp = disparity_right.at<float>(y, right_x);
            float consistency_diff = std::abs(left_disp - right_disp);
            
            if (consistency_diff > lr_threshold) {
                disparity_f32.at<float>(y, x) = -1.0f; 
                disparities.at<uchar>(y, x) = 0;
            }
        }
    }
}

void SemiGlobalMatching::sub_pixel_refinement() {
    cv::Mat refined_disp = disparity_f32.clone();
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float current_disp = disparity_f32.at<float>(y, x);
            
            if (current_disp <= 0) continue;
            
            int d = static_cast<int>(current_disp);
            
            if (d <= 0 || d >= max_disp - 1) continue;
            
            int c_minus = aggregated_cost.at<int>(y * width + x, d - 1);
            int c_center = aggregated_cost.at<int>(y * width + x, d);
            int c_plus = aggregated_cost.at<int>(y * width + x, d + 1);
            
            int denom = 2 * (c_minus - 2 * c_center + c_plus);
            if (denom != 0) {
                float delta = static_cast<float>(c_minus - c_plus) / denom;
                if (std::abs(delta) < 1.0f) {
                    refined_disp.at<float>(y, x) = d + delta;
                }
            }
        }
    }
    
    disparity_f32 = refined_disp;
    
    // update disparity map
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float disp = disparity_f32.at<float>(y, x);
            if (disp > 0) {
                disparities.at<uchar>(y, x) = static_cast<uchar>(std::min(255.0f, disp * 255.0f / max_disp));
            } else {
                disparities.at<uchar>(y, x) = 0;
            }
        }
    }
}

void SemiGlobalMatching::median_filter() {
    cv::Mat filtered;
    cv::medianBlur(disparities, filtered, 3);
    disparities = filtered;
}

bool SemiGlobalMatching::is_valid_pixel(int x, int y) const {
    return x >= 0 && x < width && y >= 0 && y < height;
}

cv::Mat SemiGlobalMatching::get_disparity() const {
    return disparities.clone();
}

cv::Mat SemiGlobalMatching::get_disparity_float() const {
    return disparity_f32.clone();
}

cv::Mat SemiGlobalMatching::get_disparity_left_right() const {
    return disparity_right.clone();
}