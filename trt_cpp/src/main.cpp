#include "TRTInfer.h"
#include "DataProcessor.h"
#include "CudaProcessor.h"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

int main() {
    const std::string engine_path = "E:/lightweight_G/checkpoint/model_trt_3060_fp16.engine";
    const std::string npy_dir     = "E:/lightweight_G/npy/"; 
    const std::string out_dir     = "E:/lightweight_G/trt_cpp/build/results/";
    
    if (!fs::exists(out_dir)) fs::create_directories(out_dir);

    TRTInfer trt;
    if (!trt.initEngine(engine_path)) {
        std::cerr << "❌ Engine 加载失败!" << std::endl;
        return -1;
    }

    // =========================================================================
    // 【核心优化：显存池策略】
    // 以前每处理一张图都要 cudaMalloc 和 cudaFree，非常卡顿。
    // 现在我们在主循环外一次性分配"最大可能需要"的显存大小，直接复用！
    // 假定你的最大散斑图像不超过 2048x2048 (如果更大请按需修改)
    // =========================================================================
    const size_t MAX_ELEMENTS = 2048 * 2048;
    float *d_raw, *d_pre, *d_trt_out, *d_post;
    
    cudaMalloc(&d_raw, MAX_ELEMENTS * sizeof(float));      // 存放：主存拷贝进来的原图
    cudaMalloc(&d_pre, MAX_ELEMENTS * sizeof(float));      // 存放：GPU预处理(Pad+BoxCox)后的图
    cudaMalloc(&d_trt_out, MAX_ELEMENTS * sizeof(float));  // 存放：TRT推理输出的结果
    cudaMalloc(&d_post, MAX_ELEMENTS * sizeof(float));     // 存放：GPU后处理(裁切+逆BoxCox)后的图

    std::cout << "📂 开始处理 NPY 文件 (全显存无缝 Pipeline 模式)..." << std::endl;

    for (const auto& entry : fs::directory_iterator(npy_dir)) {
        if (entry.path().extension() != ".npy") continue;
        
        std::string filename = entry.path().filename().string();
        int h, w;
        
        // 读取 Python 生成的测试数据
        auto raw_vec = DataProcessor::loadNpyUltimate(entry.path().string(), h, w);
        
        // 计算要求 32 对齐所需的 Padding 数量
        int pad_w = (32 - w % 32) % 32;
        int pad_h = (32 - h % 32) % 32;
        int padded_w = w + pad_w;
        int padded_h = h + pad_h;

        // 安全检查：防止加载了超大图片导致写入越界崩溃
        if (raw_vec.empty() || (padded_h * padded_w) > MAX_ELEMENTS) {
            std::cerr << "⚠️ 跳过 " << filename << " (数据为空或超出预分配显存大小)" << std::endl;
            continue;
        }

        // 把 vector 映射为 cv::Mat 方便计算全局极值
        cv::Mat raw_img(h, w, CV_32FC1, raw_vec.data());
        double raw_min, raw_max;
        cv::minMaxLoc(raw_img, &raw_min, &raw_max);

        auto t1 = std::chrono::high_resolution_clock::now();

        // -------------------------------------------------------------------
        // 🔥 全显存流水线启动 (Pipeline)
        // 整个过程中数据一直在 GPU 内部流转，完全没有耗时的返回 CPU 操作
        // -------------------------------------------------------------------

        // 步骤 1: 将原始图片从 CPU 送入 GPU 显存池 [唯一一次上传]
        cudaMemcpyAsync(d_raw, raw_img.ptr<float>(), h * w * sizeof(float), cudaMemcpyHostToDevice, 0);

        // 步骤 2: 触发预处理 Kernel (d_raw -> d_pre)
        CudaProcessor::preprocessGPU(d_raw, d_pre, w, h, pad_w, pad_h);

        // 步骤 3: 触发 TRT 推理 (d_pre -> d_trt_out)
        trt.infer_gpu(d_pre, d_trt_out, padded_h, padded_w);

        // 步骤 4: 触发后处理 Kernel (d_trt_out -> d_post)
        CudaProcessor::postprocessGPU(d_trt_out, d_post, w, h, pad_w, pad_h);

        // 步骤 5: 将最终清洗好的图从 GPU 拿回 CPU [唯一一次下载]
        cv::Mat denoised(h, w, CV_32FC1);
        cudaMemcpyAsync(denoised.ptr<float>(), d_post, h * w * sizeof(float), cudaMemcpyDeviceToHost, 0);

        // 步骤 6: 此时 GPU 才刚刚收到一系列指令，由于是异步操作，必须在这里阻塞 CPU 等待所有操作做完
        cudaStreamSynchronize(0);

        auto t2 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        // ---------------- 流水线结束 ----------------

        // 寻找去噪后图像的极值，用于对比和保存
        double pred_min, pred_max;
        cv::minMaxLoc(denoised, &pred_min, &pred_max);
        
        // 【关键逻辑】为了让对比图亮度一致，完美还原 Python 中计算全局最大值的逻辑
        float global_vmax = static_cast<float>(std::max(raw_max, pred_max));
        
        // 保存原图的可视化结果
        DataProcessor::saveVisualImage(raw_img, out_dir + filename + "_input.png", global_vmax);
        
        // 拼接左上角的文字信息，包括耗时，并保存结果图
        std::stringstream ss;
        ss << filename << " | " << w << "x" << h << " | " << std::fixed << std::setprecision(2) << ms << " ms";
        DataProcessor::saveVisualImage(denoised, out_dir + filename + "_result.png", global_vmax, ss.str());

        std::cout << "✔️ 处理成功: " << filename << " [" << w << "x" << h << "] | " << ms << " ms" << std::endl;
    }

    // 清理显存池资源，结束程序
    cudaFree(d_raw);
    cudaFree(d_pre);
    cudaFree(d_trt_out);
    cudaFree(d_post);
    trt.release();
    
    return 0;
}