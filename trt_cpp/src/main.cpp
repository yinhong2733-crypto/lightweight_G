#include "TRTInfer.h"
#include "DataProcessor.h"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <iomanip>

namespace fs = std::filesystem;

int main() {
    // 路径与超参数配置 (已更新为 E 盘新路径)
    //fp16量化engine
    const std::string engine_path = "E:/lightweight_G/checkpoint/model_trt_3060_fp16.engine";
    //不量化fp32
    //const std::string engine_path = "E:/lightweight_G/checkpoint/model_trt_3060_fp32.engine";
    const std::string npy_dir     = "E:/lightweight_G/npy/"; 
    const std::string out_dir     = "E:/lightweight_G/trt_cpp/build/results/";
    
    if (!fs::exists(out_dir)) fs::create_directories(out_dir);

    TRTInfer trt;
    if (!trt.initEngine(engine_path)) {
        std::cerr << "❌ Engine 加载失败，请检查路径: " << engine_path << std::endl;
        return -1;
    }

    std::cout << "📂 开始处理 NPY 文件 (原图推理模式)..." << std::endl;

    for (const auto& entry : fs::directory_iterator(npy_dir)) {
        if (entry.path().extension() != ".npy") continue;
        
        std::string filename = entry.path().filename().string();
        int h, w;
        
        // 1. 读取原图尺寸
        auto raw_vec = DataProcessor::loadNpyUltimate(entry.path().string(), h, w);
        if (raw_vec.empty()) continue;

        cv::Mat raw_img(h, w, CV_32FC1, raw_vec.data());

        // 2. 预处理 (不再有 Resize，保留原始细节)
        auto input = DataProcessor::preprocess(raw_img);

        // 3. 执行推理 (根据当前图片的 h 和 w 动态分配内存)
        auto t1 = std::chrono::high_resolution_clock::now();
        auto output = trt.infer(input, h, w);
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        if (!output.empty()) {
            // 4. 后处理：直接在原尺寸上生成去噪结果
            cv::Mat denoised = DataProcessor::postprocess(output, w, h);
            
            // 5. 保存预览图
            DataProcessor::saveVisualImage(raw_img, out_dir + filename + "_input.png");
            
            std::stringstream ss;
            ss << filename << " | " << w << "x" << h << " | " << std::fixed << std::setprecision(2) << ms << " ms";
            DataProcessor::saveVisualImage(denoised, out_dir + filename + "_result.png", ss.str());

            std::cout << "✔️ 处理成功: " << filename << " [" << w << "x" << h << "]" << std::endl;
        }
    }

    trt.release();
    return 0;
}