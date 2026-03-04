#include "DataProcessor.h"
#include <fstream>
#include <regex>
#include <cmath>
#include <algorithm>

std::vector<float> DataProcessor::loadNpyUltimate(const std::string& filename, int& out_h, int& out_w) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return {};

    // 校验 numpy 魔数头是否正确
    char magic[6]; file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") return {};

    unsigned char major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);

    // 解析 Header 长度信息
    uint32_t header_len = 0;
    int header_size_field = (major == 1) ? 2 : 4; 
    if (major == 1) { uint16_t tmp; file.read(reinterpret_cast<char*>(&tmp), 2); header_len = tmp; }
    else { file.read(reinterpret_cast<char*>(&header_len), 4); }

    std::string header(header_len, ' ');
    file.read(&header[0], header_len);

    // 用正则表达式提取 tuple 表示的图片尺寸 (H, W)
    std::regex shape_regex(R"(\((\d+),\s*(\d+)\))");
    std::smatch match;
    if (std::regex_search(header, match, shape_regex)) {
        out_h = std::stoi(match[1]);
        out_w = std::stoi(match[2]);
    }

    int data_start = 6 + 2 + header_size_field + header_len;
    file.seekg(data_start, std::ios::beg);

    std::vector<float> result(out_h * out_w);
    // 判断 Numpy 是 float64('f8') 还是 float32('f4')
    if (header.find("'f8'") != std::string::npos || header.find("<f8") != std::string::npos) {
        // 如果是双精度，先读取进来再强转为单精度 (float) 以对齐模型输入
        std::vector<double> temp_d(out_h * out_w);
        file.read(reinterpret_cast<char*>(temp_d.data()), out_h * out_w * sizeof(double));
        for(size_t i = 0; i < temp_d.size(); ++i) result[i] = static_cast<float>(temp_d[i]);
    } else {
        // 单精度直接拷贝
        file.read(reinterpret_cast<char*>(result.data()), out_h * out_w * sizeof(float));
    }
    return result;
}

void DataProcessor::saveVisualImage(const cv::Mat& img, const std::string& save_path, float vmax, const std::string& info_text) {
    cv::Mat vis(img.size(), CV_8U);
    
    for (int r = 0; r < img.rows; ++r) {
        const float* src = img.ptr<float>(r);
        uchar* dst = vis.ptr<uchar>(r);
        for (int c = 0; c < img.cols; ++c) {
            float val = std::max(src[c], 0.0f);
            if (vmax <= 1e-8f) {
                dst[c] = 0;
            } else {
                val = std::min(val, vmax);
                // 线性归一化到 0~255
                float out_val = (val / (vmax + 1e-8f)) * 255.0f;
                
                // 【画质对齐核心修复】
                // 之前使用的 std::round 是"四舍五入"(2.5->3)
                // 而 Python 里的 numpy.rint 是"银行家舍入/偶数舍入"(2.5->2, 3.5->4)
                // C++ 中的 std::rint() 才是严格与 Python 底层一致的！这消除了 ±1 像素的取整误差
                dst[c] = static_cast<uchar>(std::clamp(std::rint(out_val), 0.0f, 255.0f)); 
            }
        }
    }

    // 转为 BGR，用于在图上用彩色绘制信息文本
    cv::Mat vis_color;
    cv::cvtColor(vis, vis_color, cv::COLOR_GRAY2BGR);
    if (!info_text.empty()) {
        cv::putText(vis_color, info_text, cv::Point(20, 40), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite(save_path, vis_color);
}