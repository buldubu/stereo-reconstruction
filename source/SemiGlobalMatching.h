#ifndef SEMI_GLOBAL_MATCHING_H
#define SEMI_GLOBAL_MATCHING_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class SemiGlobalMatching {
public:
    SemiGlobalMatching(
        const cv::Mat& left,
        const cv::Mat& right,
        int max_disp,
        int census_size,
        int filter_size,
        int occlusion_seuil,
        int penalty_1,
        int penalty_2
    );

    void SGM_process();
    cv::Mat get_disparity() const;       
    cv::Mat get_disparity_float() const;  

private:
    int max_disp, census_size, filter_size;
    int occlusion_seuil, penalty_1, penalty_2;
    int height, width;

    cv::Mat left_I, right_I;
    cv::Mat disparities;      
    cv::Mat disparity_f32;    

    std::vector<cv::Mat> left_cost_volume;
    std::vector<std::vector<cv::Mat>> aggregation_volume;

    void compute_costs();
    void aggregate_costs();
    cv::Mat get_path_cost(const cv::Mat& block);
};

#endif 
