#pragma once
#include <opencv2/opencv.hpp>

class CudaProcessor {
public:
    /**
     * @brief 纯 GPU 预处理：包含镜像 Padding 和高精度 Box-Cox 变换
     * @note  为了消除 CPU-GPU 拷贝耗时，接口直接接收和输出显存指针 (Device Pointer)
     * @param d_src 输入原始图像的显存指针 (尺寸: raw_w * raw_h)
     * @param d_dst 输出预处理后图像的显存指针 (尺寸: (raw_w+pad_w) * (raw_h+pad_h))
     */
    static void preprocessGPU(const float* d_src, float* d_dst, int raw_w, int raw_h, int pad_w, int pad_h, float lam = 0.05f, float eps = 1e-6f);
    
    /**
     * @brief 纯 GPU 后处理：包含高精度逆 Box-Cox 变换和自动裁切 (Unpad)
     * @note  直接读取 TRT 推理输出的显存数据，处理完后存入目标显存
     * @param d_src 输入 TRT 推理结果的显存指针 (尺寸: (original_w+pad_w) * (original_h+pad_h))
     * @param d_dst 输出最终复原图像的显存指针 (尺寸: original_w * original_h)
     */
    static void postprocessGPU(const float* d_src, float* d_dst, int original_w, int original_h, int pad_w, int pad_h, float lam = 0.05f, float eps = 1e-6f);
};