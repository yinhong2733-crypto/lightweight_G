#include "CudaProcessor.h"
#include <cuda_runtime.h>
#include <math.h>

// ==========================================================
// GPU Kernel: 预处理 (Padding + Box-Cox)
// ==========================================================
__global__ void preprocessKernel(const float* src, float* dst, int raw_w, int raw_h, int pad_w, int pad_h, float lam, float eps) {
    // 计算当前线程对应的目标图像 (padded) 的二维坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int padded_w = raw_w + pad_w;
    int padded_h = raw_h + pad_h;

    // 越界检查
    if (x < padded_w && y < padded_h) {
        int src_x = x;
        int src_y = y;
        
        // 镜像反射 Padding 逻辑 (Reflection Pad)
        // 比如 raw_w=100, x=101 时，被映射回原始图像的对应位置
        if (src_x >= raw_w) src_x = 2 * raw_w - 2 - src_x;
        if (src_y >= raw_h) src_y = 2 * raw_h - 2 - src_y;

        // 从原始图像显存读取像素值
        float val = src[src_y * raw_w + src_x];
        
        // Box-Cox 变换前置处理：防止出现负数或 0，确保底数 > 0
        val = fmaxf(val, 0.0f) + 1.0f + eps; 
        
        // 核心 Box-Cox 变换公式写入目标显存
        dst[y * padded_w + x] = (powf(val, lam) - 1.0f) / lam;
    }
}

// ==========================================================
// GPU Kernel: 后处理 (逆 Box-Cox + Unpad)
// ==========================================================
__global__ void postprocessKernel(const float* src, float* dst, int raw_w, int raw_h, int pad_w, int pad_h, float lam, float eps) {
    // 此处的 x, y 代表裁切后(恢复原始尺寸)的图像坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int padded_w = raw_w + pad_w;

    // 只处理有效区域，天然实现了自动裁切去 Padding
    if (x < raw_w && y < raw_h) {
        // 从 TRT 推理输出的显存中读取对应位置的像素
        float val = src[y * padded_w + x];
        
        // 逆 Box-Cox 变换公式：u = lambda * y + 1
        float u = lam * val + 1.0f;
        u = fmaxf(u, eps); // 防止底数为负数导致 powf 异常
        
        // 算出原始像素值：x = u^(1/lambda) - 1，写入目标显存
        dst[y * raw_w + x] = powf(u, 1.0f / lam) - 1.0f;
    }
}

// ==========================================================
// 主机端启动函数 (Host Functions)
// ==========================================================
void CudaProcessor::preprocessGPU(const float* d_src, float* d_dst, int raw_w, int raw_h, int pad_w, int pad_h, float lam, float eps) {
    int padded_w = raw_w + pad_w;
    int padded_h = raw_h + pad_h;
    
    // 设置线程块大小 (16x16 = 256 线程，对多数 GPU 都是较好的平衡)
    dim3 block(16, 16);
    // 动态计算 Grid 大小，确保覆盖整张图像
    dim3 grid((padded_w + block.x - 1) / block.x, (padded_h + block.y - 1) / block.y);
    
    // 启动预处理核函数，这里使用的是默认流 (Stream 0)
    preprocessKernel<<<grid, block>>>(d_src, d_dst, raw_w, raw_h, pad_w, pad_h, lam, eps);
}

void CudaProcessor::postprocessGPU(const float* d_src, float* d_dst, int original_w, int original_h, int pad_w, int pad_h, float lam, float eps) {
    dim3 block(16, 16);
    dim3 grid((original_w + block.x - 1) / block.x, (original_h + block.y - 1) / block.y);
    
    // 启动后处理核函数
    postprocessKernel<<<grid, block>>>(d_src, d_dst, original_w, original_h, pad_w, pad_h, lam, eps);
}