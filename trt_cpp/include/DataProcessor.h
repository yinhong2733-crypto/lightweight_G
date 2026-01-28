#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class DataProcessor {
public:
    static std::vector<float> loadNpyUltimate(const std::string& filename, int& out_h, int& out_w);
    static std::vector<float> preprocess(const cv::Mat& raw_img, int target_size = 680);
    static cv::Mat postprocess(const std::vector<float>& infer_out, int orig_w, int orig_h, int model_size = 680);
    static void saveVisualImage(const cv::Mat& raw_float_img, const std::string& save_path);
};