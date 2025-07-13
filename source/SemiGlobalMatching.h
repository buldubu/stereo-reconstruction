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
        int max_disp = 128,
        int census_size = 5,
        int penalty_1 = 8,
        int penalty_2 = 100,
        float lr_threshold = 1.0f
    );

    void SGM_process();
    cv::Mat get_disparity() const;       
    cv::Mat get_disparity_float() const;  
    cv::Mat get_disparity_left_right() const;

private:
    int max_disp, census_size;
    int penalty_1, penalty_2;
    int height, width;
    float lr_threshold;

    cv::Mat left_I, right_I;
    cv::Mat disparities;      
    cv::Mat disparity_f32;
    cv::Mat disparity_right; 
    
    cv::Mat left_census, right_census;
    
    cv::Mat cost_volume; 
    cv::Mat aggregated_cost; 
    
    static const int NUM_PATHS = 8;
    const int dir_x[NUM_PATHS] = {0, 1, 1, 1, 0, -1, -1, -1};
    const int dir_y[NUM_PATHS] = {-1, -1, 0, 1, 1, 1, 0, -1};

    void compute_census_transform();
    void compute_matching_cost();
    void aggregate_costs();
    void compute_disparity_WTA();
    void compute_right_disparity();
    void left_right_consistency_check();
    void sub_pixel_refinement();
    void median_filter();
    
    // Helper funcs
    uint64_t compute_census_at(const cv::Mat& img, int x, int y) const;
    int hamming_distance(uint64_t a, uint64_t b) const;
    void aggregate_path(int start_x, int start_y, int dx, int dy);
    bool is_valid_pixel(int x, int y) const;
};

#endif