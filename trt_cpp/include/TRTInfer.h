#pragma once
#include <NvInfer.h>
#include <string>
#include <vector>
#include <iostream>

// 自定义日志打印器，用于捕获 TensorRT 内部的 Warning 和 Error 信息
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // 只打印 Warning 及以上级别的日志，忽略海量的 Info 日志
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

class TRTInfer {
public:
    // 初始化并反序列化 Engine 文件
    bool initEngine(const std::string& engine_path);
    
    // 兼容老版本的 CPU 数据接口 (不推荐在对性能要求高的循环中使用)
    std::vector<float> infer(const std::vector<float>& input_data, int h, int w);
    
    /**
     * @brief 【核心优化】纯 GPU 推理接口
     * @note  传入传出皆为预先分配好的显存指针，彻底省去了内存到显存的往返拷贝
     * @param d_in  预处理后的输入显存指针
     * @param d_out 推理结果的输出显存指针
     */
    bool infer_gpu(void* d_in, void* d_out, int h, int w);
    
    // 释放 TRT 占用的相关资源
    void release();
    
private:
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    Logger logger;
};