#pragma once
#include <NvInfer.h>
#include <string>
#include <vector>
#include <iostream>

// TensorRT 日志接收器：捕获并打印模型运行时的错误或警告
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class TRTInfer {
public:
    // 加载序列化后的 .engine 文件
    bool initEngine(const std::string& engine_path);
    // 执行推理计算
    std::vector<float> infer(const std::vector<float>& input_data, int input_h, int input_w);
    // 释放显存和资源
    void release();

private:
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    Logger logger;
};