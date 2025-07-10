#include "SemiGlobalMatching.h"
#include <limits>

SemiGlobalMatching::SemiGlobalMatching(
    const cv::Mat& left, const cv::Mat& right,
    int max_disp_, int census_size_,
    int filter_size_, int occlusion_threshold_,
    int penalty1_, int penalty2_)
  : max_disp(max_disp_),
    half_census(census_size_/2),
    filter_size(filter_size_),
    occl_thresh(occlusion_threshold_),
    P1(penalty1_), P2(penalty2_)
{

    cv::cvtColor(left,  left_gray,  cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
    H = left_gray.rows;  W = left_gray.cols;

    censusL = cv::Mat(H, W, CV_64F);
    censusR = cv::Mat(H, W, CV_64F);
    costVol.resize(max_disp, cv::Mat(H, W, CV_16U));
    aggVol.assign(4, std::vector<cv::Mat>(max_disp, cv::Mat(H, W, CV_32S)));
    dispRaw = cv::Mat(H, W, CV_32F, cv::Scalar(-1));
    disp8   = cv::Mat(H, W, CV_8U,  cv::Scalar(0));
}

void SemiGlobalMatching::compute() {
    censusTransform(left_gray,  censusL);
    censusTransform(right_gray, censusR);
    buildCostVolume();
    aggregateCosts();
    selectDisparity();

    SemiGlobalMatching rightSGM = *this;
    std::swap(rightSGM.censusL, rightSGM.censusR);
    rightSGM.buildCostVolume();
    rightSGM.aggregateCosts();
    rightSGM.selectDisparity();
    leftRightCheck(rightSGM.dispRaw);

    holeFillingAndFilter();

    dispRaw.convertTo(disp8, CV_8U, 255.0f / max_disp);
}


uint64_t SemiGlobalMatching::censusAt(const cv::Mat& I, int y, int x) {
    uint64_t code = 0, bit = 1;
    uchar center = I.at<uchar>(y, x);
    for(int dy=-half_census; dy<=half_census; ++dy) {
        for(int dx=-half_census; dx<=half_census; ++dx) {
            if(dy==0 && dx==0) continue;
            if(I.at<uchar>(y+dy, x+dx) < center)
                code |= bit;
            bit <<= 1;
        }
    }
    return code;
}

void SemiGlobalMatching::censusTransform(const cv::Mat& src, cv::Mat& dst) {
    for(int y=half_census; y<H-half_census; ++y)
        for(int x=half_census; x<W-half_census; ++x)
            dst.at<double>(y, x) = static_cast<double>(censusAt(src, y, x));
}

void SemiGlobalMatching::buildCostVolume() {
    for(int d=0; d<max_disp; ++d) {
        auto& C = costVol[d];
        C.setTo(USHRT_MAX);
        for(int y=half_census; y<H-half_census; ++y) {
            for(int x=half_census; x<W-half_census; ++x) {
                int xr = x - d;
                if(xr < half_census) continue;
                uint64_t L = (uint64_t)censusL.at<double>(y,x);
                uint64_t R = (uint64_t)censusR.at<double>(y,xr);
                C.at<uint16_t>(y,x) = static_cast<uint16_t>(popcount64(L ^ R));
            }
        }
    }
}

cv::Mat SemiGlobalMatching::aggregate1D(const cv::Mat& slice) {
    int N = slice.rows;  
    cv::Mat path(N, max_disp, CV_32S);

    for(int d=0; d<max_disp; ++d)
        path.at<int>(0,d) = slice.at<int>(0,d);

    // İteratif
    for(int i=1; i<N; ++i) {
        int prev_min = INT_MAX;
        for(int d=0; d<max_disp; ++d)
            prev_min = std::min(prev_min, path.at<int>(i-1,d));

        for(int d=0; d<max_disp; ++d) {
            int C = slice.at<int>(i,d);
            int c1 = path.at<int>(i-1,d);
            int c2 = (d>0   ? path.at<int>(i-1,d-1) + P1 : INT_MAX);
            int c3 = (d<max_disp-1 ? path.at<int>(i-1,d+1) + P1 : INT_MAX);
            int c4 = prev_min + P2;
            int m  = std::min({c1,c2,c3,c4});
            path.at<int>(i,d) = C + m - prev_min;
        }
    }
    return path;
}

void SemiGlobalMatching::aggregate_costs() {
    for(int x = 0; x < width; ++x) {
        cv::Mat col(height, max_disp, CV_32S);
        for(int y = 0; y < height; ++y)
            for(int d = 0; d < max_disp; ++d)
                col.at<int>(y,d) = left_cost_volume[d].at<int>(y,x);

        cv::Mat north = get_path_cost(col);

        cv::Mat col_flip, south_tmp, south;
        cv::flip(col, col_flip, 0);
        south_tmp = get_path_cost(col_flip);
        cv::flip(south_tmp, south, 0);

        for(int y = 0; y < height; ++y) {
            for(int d = 0; d < max_disp; ++d) {
                aggregation_volume[0][d].at<int>(y,x) = north.at<int>(y,d);
                aggregation_volume[1][d].at<int>(y,x) = south.at<int>(y,d);
            }
        }
    }

    for(int y = 0; y < height; ++y) {
        cv::Mat row(width, max_disp, CV_32S);
        for(int x = 0; x < width; ++x)
            for(int d = 0; d < max_disp; ++d)
                row.at<int>(x,d) = left_cost_volume[d].at<int>(y,x);

        cv::Mat west = get_path_cost(row);

        cv::Mat row_flip, east_tmp, east;
        cv::flip(row, row_flip, 0);
        east_tmp = get_path_cost(row_flip);
        cv::flip(east_tmp, east, 0);

        for(int x = 0; x < width; ++x) {
            for(int d = 0; d < max_disp; ++d) {
                aggregation_volume[2][d].at<int>(y,x) = west.at<int>(x,d);
                aggregation_volume[3][d].at<int>(y,x) = east.at<int>(x,d);
            }
        }
    }
}
void SemiGlobalMatching::selectDisparity() {

    for(int y=0; y<H; ++y) {
        for(int x=0; x<W; ++x) {
            int bestD = 0;
            int bestCost = INT_MAX;
            for(int d=0; d<max_disp; ++d) {
                int c = 0;
                for(int dir=0; dir<4; ++dir)
                    c += aggVol[dir][d].at<int>(y,x);
                if(c < bestCost) {
                    bestCost = c;
                    bestD = d;
                }
            }
            dispRaw.at<float>(y,x) = static_cast<float>(bestD);
        }
    }
}

void SemiGlobalMatching::leftRightCheck(const cv::Mat& rightRaw) {
    for(int y=0; y<H; ++y){
        for(int x=0; x<W; ++x){
            float dl = dispRaw.at<float>(y,x);
            int xr = int(x - dl);
            if(xr<0 || xr>=W) { dispRaw.at<float>(y,x) = -1; continue; }
            float dr = rightRaw.at<float>(y,xr);
            if(std::abs(dl - dr) > occl_thresh)
                dispRaw.at<float>(y,x) = -1;
        }
    }
}

void SemiGlobalMatching::holeFillingAndFilter() {
    for(int y=0; y<H; ++y){
        for(int x=0; x<W; ++x){
            if(dispRaw.at<float>(y,x) < 0) {
                float sum=0; int cnt=0;
                for(int dy=-1; dy<=1; ++dy){
                    for(int dx=-1; dx<=1; ++dx){
                        int yy=y+dy, xx=x+dx;
                        if(yy<0||yy>=H||xx<0||xx>=W) continue;
                        float v = dispRaw.at<float>(yy,xx);
                        if(v>=0) { sum+=v; cnt++; }
                    }
                }
                if(cnt>0) dispRaw.at<float>(y,x) = sum/cnt;
            }
        }
    }
    cv::medianBlur(disp8, disp8, filter_size);
}

cv::Mat SemiGlobalMatching::getDisparity8() const {
    return disp8.clone();
}

cv::Mat SemiGlobalMatching::getDisparityRaw() const {
    return dispRaw.clone();
}
