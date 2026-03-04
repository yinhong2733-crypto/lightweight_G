#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class DataProcessor {
public:
    // 解析标准的 Numpy (.npy) 二进制文件，支持 float32 和 float64 自动转换
    static std::vector<float> loadNpyUltimate(const std::string& filename, int& out_h, int& out_w);
    
    // 将浮点矩阵映射为可视化图像 (CV_8U) 并附加文字信息
    // ⚠️ 画质对齐注意：此方法用于生成预览图。若要计算 PSNR，请另外保存一份浮点矩阵
    static void saveVisualImage(const cv::Mat& img, const std::string& save_path, float vmax, const std::string& info_text = "");
};