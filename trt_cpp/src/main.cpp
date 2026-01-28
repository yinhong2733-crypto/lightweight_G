#include "TRTInfer.h"
#include "DataProcessor.h"
#include <iostream>
#include <filesystem>
#include <chrono> // 引入计时库

namespace fs = std::filesystem;

int main() {
    // ================= 配置路径 =================
    const std::string engine_path = "C:/Users/yinhong/Desktop/lightweight_G/checkpoint/model_2_trt_2060_fp16.engine";
    const std::string npy_path    = "C:/Users/yinhong/Desktop/lightweight_G/npy/0.npy";
    const std::string out_dir     = "C:/Users/yinhong/Desktop/lightweight_G/trt_cpp/build/results/";
    
    // 你的模型输入尺寸 (与 torch.randn(1, 1, 680, 680) 对应)
    const int MODEL_SIZE = 680; 

    // 确保输出目录存在
    if (!fs::exists(out_dir)) fs::create_directories(out_dir);

    // ================= 1. 初始化 TensorRT 引擎 =================
    TRTInfer trt;
    if (!trt.initEngine(engine_path)) {
        std::cerr << "❌ 引擎初始化失败！" << std::endl;
        return -1;
    }

    // ================= 2. 读取 NPY 数据 (已验证成功) =================
    int orig_h, orig_w;
    std::vector<float> raw_data = DataProcessor::loadNpyUltimate(npy_path, orig_h, orig_w);
    
    if (raw_data.empty()) {
        std::cerr << "❌ NPY 读取失败" << std::endl;
        return -1;
    }
    
    // 构造 OpenCV 矩阵方便处理
    cv::Mat raw_img(orig_h, orig_w, CV_32FC1, raw_data.data());

    // 打印数据范围确认无误
    double minV, maxV;
    cv::minMaxLoc(raw_img, &minV, &maxV);
    printf("📊 [数据检查] 原始范围: [%.4f, %.4f], 尺寸: %dx%d\n", minV, maxV, orig_w, orig_h);

    // [新增] 保存原始图片 (归一化到 0-255 可视化)
    std::string original_save_path = out_dir + "original_input.png";
    DataProcessor::saveVisualImage(raw_img, original_save_path);
    std::cout << "💾 [保存] 原始图片已保存至: " << original_save_path << std::endl;

    // ================= 3. 预处理 =================
    // 缩放至 680x680 -> Log 变换
    std::vector<float> input_tensor = DataProcessor::preprocess(raw_img, MODEL_SIZE);

    // ================= 4. 推理 + 计时 (关键步骤) =================
    std::cout << "🚀 [推理] 开始执行 TensorRT 推理..." << std::endl;

    // --- 计时开始 ---
    auto start_time = std::chrono::high_resolution_clock::now();

    // 执行推理
    std::vector<float> output_tensor = trt.infer(input_tensor, MODEL_SIZE, MODEL_SIZE);

    // --- 计时结束 ---
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // 计算耗时 (毫秒)
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "⏱️ [性能] 推理耗时: " << duration << " ms" << std::endl;

    // ================= 5. 后处理与结果保存 =================
    if (!output_tensor.empty()) {
        // Exp 变换 -> 还原回原始尺寸 (orig_w, orig_h)
        cv::Mat final_res = DataProcessor::postprocess(output_tensor, orig_w, orig_h, MODEL_SIZE);
        
        std::string denoised_save_path = out_dir + "denoised_result.png";
        cv::imwrite(denoised_save_path, final_res);
        
        std::cout << "✅ [完成] 去噪图片已保存至: " << denoised_save_path << std::endl;
    } else {
        std::cerr << "❌ 推理返回空数据！" << std::endl;
    }

    trt.release();
    return 0;
}