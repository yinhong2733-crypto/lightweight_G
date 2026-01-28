#include "TRTInfer.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>

bool TRTInfer::initEngine(const std::string& engine_path) {
    release();
    // 1. 以二进制模式读取引擎文件
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开引擎文件: " << engine_path << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(file_size);
    file.read(engine_data.data(), file_size);
    file.close();

    // 2. 核心：反序列化引擎
    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engine_data.data(), file_size);
    if (!engine) return false;

    // 3. 创建执行上下文（类似于启动一个计算线程）
    context = engine->createExecutionContext();
    return true;
}

std::vector<float> TRTInfer::infer(const std::vector<float>& input_data, int h, int w) {
    std::vector<float> output_data;
    if (!context || !engine) return output_data;

    // 获取输入/输出绑定（0号通常是Input, 1号通常是Output）
    int nbBindings = engine->getNbBindings();
    std::vector<void*> device_ptrs(nbBindings, nullptr);
    std::vector<size_t> binding_sizes(nbBindings, 0);

    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; j++) {
            size *= (dims.d[j] > 0) ? dims.d[j] : 1;
        }
        binding_sizes[i] = size * sizeof(float);
        // 在显卡（GPU）上分配内存
        cudaMalloc(&device_ptrs[i], binding_sizes[i]);
    }

    // A. CPU -> GPU: 拷贝输入数据
    cudaMemcpy(device_ptrs[0], input_data.data(), binding_sizes[0], cudaMemcpyHostToDevice);
    
    // B. GPU 计算: 执行推理
    context->executeV2(device_ptrs.data());
    
    // C. GPU -> CPU: 拷回处理结果
    output_data.resize(binding_sizes[1] / sizeof(float));
    cudaMemcpy(output_data.data(), device_ptrs[1], binding_sizes[1], cudaMemcpyDeviceToHost);

    // D. 清理本次推理申请的临时显存
    for (void* ptr : device_ptrs) cudaFree(ptr);
    return output_data;
}

void TRTInfer::release() {
    if (context) context->destroy();
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();
    context = nullptr; engine = nullptr; runtime = nullptr;
}