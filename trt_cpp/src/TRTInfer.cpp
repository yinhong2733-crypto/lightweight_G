#include "TRTInfer.h"
#include <cuda_runtime_api.h>
#include <fstream>

bool TRTInfer::initEngine(const std::string& engine_path) {
    release();
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.is_open()) return false;
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), size);

    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(data.data(), size);
    if (!engine) return false;
    context = engine->createExecutionContext();
    return true;
}

std::vector<float> TRTInfer::infer(const std::vector<float>& input_data, int h, int w) {
    if (!context || !engine) return {};

    const char* in_name = "input";
    const char* out_name = "output";

    // 设置动态输入的实际尺寸，这决定了本次推理的计算量
    nvinfer1::Dims dims = engine->getTensorShape(in_name);
    dims.d[0] = 1; dims.d[1] = 1; dims.d[2] = h; dims.d[3] = w;
    context->setInputShape(in_name, dims);

    size_t total_elements = 1 * 1 * h * w;
    void *d_in, *d_out;
    cudaMalloc(&d_in, total_elements * sizeof(float));
    cudaMalloc(&d_out, total_elements * sizeof(float));

    // 绑定显存地址
    context->setTensorAddress(in_name, d_in);
    context->setTensorAddress(out_name, d_out);

    cudaMemcpy(d_in, input_data.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    context->enqueueV3(0); 
    cudaStreamSynchronize(0); // 必须同步以确保结果已写入显存

    std::vector<float> results(total_elements);
    cudaMemcpy(results.data(), d_out, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    return results;
}

void TRTInfer::release() {
    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;
    context = nullptr; engine = nullptr; runtime = nullptr;
}