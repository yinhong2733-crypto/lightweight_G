#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class DataProcessor {
public:
    // 读取 NPY 文件并自动获取其原始高度 (out_h) 和宽度 (out_w)
    static std::vector<float> loadNpyUltimate(const std::string& filename, int& out_h, int& out_w);
    
    // 预处理：执行 Box-Cox 变换。不再进行任何 Resize 缩放
    static std::vector<float> preprocess(const cv::Mat& raw_img, float lam = 0.05f, float eps = 1e-6f);
    
    // 后处理：执行反 Box-Cox 变换，并利用 MINMAX 归一化确保显色正常
    static cv::Mat postprocess(const std::vector<float>& infer_out, int w, int h, float lam = 0.05f, float eps = 1e-6f);
    
    // 图像保存：支持在图像左上角绘制推理时间、分辨率等信息
    static void saveVisualImage(const cv::Mat& img, const std::string& save_path, const std::string& info_text = "");
};