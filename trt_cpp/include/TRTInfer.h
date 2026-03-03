#pragma once
#include <NvInfer.h>
#include <string>
#include <vector>
#include <iostream>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

class TRTInfer {
public:
    bool initEngine(const std::string& engine_path);
    std::vector<float> infer(const std::vector<float>& input_data, int h, int w);
    void release();
private:
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    Logger logger;
};