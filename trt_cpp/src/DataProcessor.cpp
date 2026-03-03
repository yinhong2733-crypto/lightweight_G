#include "DataProcessor.h"
#include <fstream>
#include <regex>

// NPY 读取：通过解析文件头获取准确的 shape，并使用 seekg 定位数据区
std::vector<float> DataProcessor::loadNpyUltimate(const std::string& filename, int& out_h, int& out_w) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return {};

    char magic[6]; file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") return {};

    unsigned char major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    int header_size_field = (major == 1) ? 2 : 4; 
    if (major == 1) { uint16_t tmp; file.read(reinterpret_cast<char*>(&tmp), 2); header_len = tmp; }
    else { file.read(reinterpret_cast<char*>(&header_len), 4); }

    std::string header(header_len, ' ');
    file.read(&header[0], header_len);

    std::regex shape_regex(R"(\((\d+),\s*(\d+)\))");
    std::smatch match;
    if (std::regex_search(header, match, shape_regex)) {
        out_h = std::stoi(match[1]);
        out_w = std::stoi(match[2]);
    }

    // 跳转到数据起始位置
    int data_start = 6 + 2 + header_size_field + header_len;
    file.seekg(data_start, std::ios::beg);

    std::vector<float> result(out_h * out_w);
    if (header.find("'f8'") != std::string::npos || header.find("<f8") != std::string::npos) {
        std::vector<double> temp_d(out_h * out_w);
        file.read(reinterpret_cast<char*>(temp_d.data()), out_h * out_w * sizeof(double));
        for(size_t i = 0; i < temp_d.size(); ++i) result[i] = static_cast<float>(temp_d[i]);
    } else {
        file.read(reinterpret_cast<char*>(result.data()), out_h * out_w * sizeof(float));
    }
    return result;
}

// 预处理：不再 Resize，直接对原图进行像素级 Box-Cox 变换
std::vector<float> DataProcessor::preprocess(const cv::Mat& raw_img, float lam, float eps) {
    cv::Mat u, y;
    cv::max(raw_img, 0.0f, raw_img); // 修正可能存在的负值
    cv::add(raw_img, 1.0f + eps, u); 
    cv::pow(u, lam, y);
    cv::subtract(y, 1.0f, y);
    cv::divide(y, lam, y);
    
    // 返回一维向量供 TensorRT 使用
    if (y.isContinuous()) {
        return std::vector<float>((float*)y.data, (float*)y.data + y.total());
    } else {
        cv::Mat cont = y.clone();
        return std::vector<float>((float*)cont.data, (float*)cont.data + cont.total());
    }
}

// 后处理：执行反向变换，并使用 NORM_MINMAX 归一化解决黑图问题
cv::Mat DataProcessor::postprocess(const std::vector<float>& infer_out, int w, int h, float lam, float eps) {
    cv::Mat y(h, w, CV_32FC1, (void*)infer_out.data());
    cv::Mat u, x, final_8u;

    cv::multiply(y, lam, u);
    cv::add(u, 1.0f, u);
    cv::max(u, eps, u); 
    cv::pow(u, 1.0f / lam, x);
    cv::subtract(x, 1.0f + eps, x);

    // 自动将图像拉伸到 0-255 亮度范围
    cv::normalize(x, final_8u, 0, 255, cv::NORM_MINMAX);
    final_8u.convertTo(final_8u, CV_8U);
    return final_8u;
}

void DataProcessor::saveVisualImage(const cv::Mat& img, const std::string& save_path, const std::string& info_text) {
    cv::Mat vis;
    if (img.depth() != CV_8U) {
        cv::normalize(img, vis, 0, 255, cv::NORM_MINMAX);
        vis.convertTo(vis, CV_8U);
    } else {
        vis = img.clone();
    }

    cv::Mat vis_color;
    cv::cvtColor(vis, vis_color, cv::COLOR_GRAY2BGR);
    if (!info_text.empty()) {
        cv::putText(vis_color, info_text, cv::Point(20, 40), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite(save_path, vis_color);
}