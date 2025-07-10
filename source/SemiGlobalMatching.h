#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>

class SemiGlobalMatching {
public:
    SemiGlobalMatching(const cv::Mat& left,
                       const cv::Mat& right,
                       int max_disp = 64,
                       int census_size = 9,
                       int filter_size = 5,
                       int occlusion_seuil = 1,
                       int penalty_1 = 10,
                       int penalty_2 = 120);

    void SGM_process();

    cv::Mat get_disparity() const;

    // Raw float disparity (0 - max_disp)
    cv::Mat get_disparity_float() const;

private:
    int max_disp;
    int census_size;
    int filter_size;
    int occlusion_seuil;
    int penalty_1;
    int penalty_2;

    cv::Mat left_I;
    cv::Mat right_I;
    int height;
    int width;

    std::vector<cv::Mat> left_cost_volume;
    std::vector<std::vector<cv::Mat>> aggregation_volume;

    cv::Mat disparities;
    cv::Mat disparity_f32;

    void compute_costs();
    void aggregate_costs();
};
