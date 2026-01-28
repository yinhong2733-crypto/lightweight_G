#include "DataProcessor.h"
#include <fstream>
#include <iostream>
#include <regex>

// 1. 读取 NPY (包含之前的修复)
std::vector<float> DataProcessor::loadNpyUltimate(const std::string& filename, int& out_h, int& out_w) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return {};

    char magic[6]; file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") return {};

    unsigned char major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    int header_size_field = (major == 1) ? 2 : 4; // 计算长度字段本身占用的字节数

    if (major == 1) {
        uint16_t tmp; file.read(reinterpret_cast<char*>(&tmp), 2);
        header_len = tmp;
    } else {
        file.read(reinterpret_cast<char*>(&header_len), 4);
    }

    std::string header(header_len, ' ');
    file.read(&header[0], header_len);

    std::regex shape_regex(R"(\((\d+),\s*(\d+)\))");
    std::smatch match;
    if (std::regex_search(header, match, shape_regex)) {
        out_h = std::stoi(match[1]);
        out_w = std::stoi(match[2]);
    }

    // 绝对定位数据区：Magic(6) + Ver(2) + LenField(2 or 4) + HeaderLen
    int data_start = 6 + 2 + header_size_field + header_len;
    file.seekg(data_start, std::ios::beg);

    std::vector<float> result(out_h * out_w);
    // 判断数据类型是否为 double ('f8')
    if (header.find("'f8'") != std::string::npos || header.find("<f8") != std::string::npos) {
        std::vector<double> temp_d(out_h * out_w);
        file.read(reinterpret_cast<char*>(temp_d.data()), out_h * out_w * sizeof(double));
        for(size_t i = 0; i < temp_d.size(); ++i) result[i] = static_cast<float>(temp_d[i]);
    } else {
        file.read(reinterpret_cast<char*>(result.data()), out_h * out_w * sizeof(float));
    }
    return result;
}

// 2. 预处理 (Resize -> Log)
std::vector<float> DataProcessor::preprocess(const cv::Mat& raw_img, int target_size) {
    cv::Mat resized, log_img;
    cv::resize(raw_img, resized, cv::Size(target_size, target_size));
    cv::max(resized, 0.0f, resized); // 去负值
    cv::log(resized + 1.0f, log_img); // Log1p
    
    // 转换为 float 向量
    if (log_img.isContinuous()) {
        return std::vector<float>((float*)log_img.data, (float*)log_img.data + target_size * target_size);
    } else {
        // 防止非连续内存（虽然resize后通常是连续的）
        std::vector<float> vec; 
        log_img = log_img.reshape(1, 1);
        log_img.copyTo(vec);
        return vec;
    }
}

// 3. 后处理 (Exp -> Resize -> Normalize 0-255)
cv::Mat DataProcessor::postprocess(const std::vector<float>& infer_out, int orig_w, int orig_h, int model_size) {
    cv::Mat out_log(model_size, model_size, CV_32FC1, (void*)infer_out.data());
    cv::Mat res_small;
    
    cv::exp(out_log, res_small);
    res_small -= 1.0f; // 还原 Log1p

    cv::Mat res_big;
    cv::resize(res_small, res_big, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_CUBIC);

    // 归一化并转为 8位图像保存
    cv::Mat final_8u;
    cv::normalize(res_big, final_8u, 0, 255, cv::NORM_MINMAX);
    final_8u.convertTo(final_8u, CV_8U);
    return final_8u;
}

// 4. 可视化辅助保存 (用于保存原图)
void DataProcessor::saveVisualImage(const cv::Mat& raw_float_img, const std::string& save_path) {
    cv::Mat vis_8u;
    // 将 float 数据归一化到 0-255 之间以便查看
    cv::normalize(raw_float_img, vis_8u, 0, 255, cv::NORM_MINMAX);
    vis_8u.convertTo(vis_8u, CV_8U);
    cv::imwrite(save_path, vis_8u);
}