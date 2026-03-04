#include "TRTInfer.h"
#include <cuda_runtime_api.h>
#include <fstream>

bool TRTInfer::initEngine(const std::string& engine_path) {
    release(); // 防止重复初始化导致内存泄漏

    // 以二进制方式读取 engine 模型文件
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.is_open()) return false;
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), size);

    // 创建运行时并反序列化模型
    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(data.data(), size);
    if (!engine) return false;
    
    // 创建执行上下文，用于具体的推理操作
    context = engine->createExecutionContext();
    return true;
}

std::vector<float> TRTInfer::infer(const std::vector<float>& input_data, int h, int w) {
    if (!context || !engine) return {};

    const char* in_name = "input";
    const char* out_name = "output";

    // 设置动态输入的实际尺寸 (Dynamic Shape)
    nvinfer1::Dims dims = engine->getTensorShape(in_name);
    dims.d[0] = 1; dims.d[1] = 1; dims.d[2] = h; dims.d[3] = w;
    context->setInputShape(in_name, dims);

    size_t total_elements = 1 * 1 * h * w;
    void *d_in, *d_out;
    
    // ⚠️ 注意：这里频繁分配和释放显存是非常耗时的操作
    cudaMalloc(&d_in, total_elements * sizeof(float));
    cudaMalloc(&d_out, total_elements * sizeof(float));

    context->setTensorAddress(in_name, d_in);
    context->setTensorAddress(out_name, d_out);

    cudaMemcpy(d_in, input_data.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);
    context->enqueueV3(0); 
    cudaStreamSynchronize(0); 

    std::vector<float> results(total_elements);
    cudaMemcpy(results.data(), d_out, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    return results;
}

bool TRTInfer::infer_gpu(void* d_in, void* d_out, int h, int w) {
    if (!context || !engine) return false;

    // 假定你的 ONNX 模型输入输出节点名字叫 input 和 output
    const char* in_name = "input";
    const char* out_name = "output";

    // 动态调整当前帧的 Tensor 尺寸
    nvinfer1::Dims dims = engine->getTensorShape(in_name);
    dims.d[0] = 1; dims.d[1] = 1; dims.d[2] = h; dims.d[3] = w;
    context->setInputShape(in_name, dims);

    // 绑定外部传入的显存地址，告诉 TRT 去哪里拿数据，算完放哪里
    context->setTensorAddress(in_name, d_in);
    context->setTensorAddress(out_name, d_out);

    // 纯异步推理提交 ( enqueueV3 是 TRT10 推荐的新 API )
    // ⚠️ 注意：这里不调用 cudaStreamSynchronize，我们将同步任务交给了外层的主循环，以此来打通流水线
    context->enqueueV3(0); 
    return true;
}

void TRTInfer::release() {
    // 按照与创建相反的顺序释放资源
    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;
    context = nullptr; engine = nullptr; runtime = nullptr;
}