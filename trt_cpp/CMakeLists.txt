cmake_minimum_required(VERSION 3.18)

# 【重要】强制指定 CUDA 12.8 编译器路径，必须放在 project 命令之前
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe")

# 开启 C++ 和 CUDA 联合编译支持
project(trt_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
# 设定默认构建类型为 Release
set(CMAKE_BUILD_TYPE Release)

# 【性能关键】指定 RTX 3060 的专属安培(Ampere)架构算力为 86，榨干显卡性能
set(CMAKE_CUDA_ARCHITECTURES 86)

# 开启 CUDA 硬件级快速数学指令，极大加速 powf, expf, sinf 等计算
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")

# 第三方库路径配置 (请确保这些路径与你电脑实际路径一致)
set(TRT_DIR "D:/TensorRT-10.15.1.29")
set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
set(OCV_DIR "D:/opencv/build")

# TensorRT 引擎文件路径
set(ENGINE_SOURCE_PATH "${PROJECT_SOURCE_DIR}/../checkpoint/model_trt_3060_fp16.engine")

# 搜集 src 目录下所有的 .cpp 和 .cu 文件参与编译
file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
add_executable(trt_denoise_show ${SRC_FILES})

# 包含头文件目录
target_include_directories(trt_denoise_show PRIVATE
    "${TRT_DIR}/include"
    "${PROJECT_SOURCE_DIR}/include"
    "${CUDA_DIR}/include"
    "${OCV_DIR}/include"
)

# 链接静态/动态库
target_link_libraries(trt_denoise_show PRIVATE
    "${TRT_DIR}/lib/nvinfer_10.lib"
    "${TRT_DIR}/lib/nvinfer_plugin_10.lib"
    "${TRT_DIR}/lib/nvinfer_dispatch_10.lib"
    "${CUDA_DIR}/lib/x64/cudart.lib"       # CUDA 运行时库
    "${CUDA_DIR}/lib/x64/cuda.lib"         # CUDA 驱动库
    "${OCV_DIR}/x64/vc16/lib/opencv_world481.lib" # OpenCV 库
)

# 【核心修复】去掉了硬编码的 /MD，避免与 Debug 模式冲突。同时用生成器表达式隔离 MSVC 参数。
if(MSVC)
    target_compile_options(trt_denoise_show PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:/W3;/utf-8;/wd4996>
    )
endif()

# 编译完成后，自动将必需的 dll 文件和 engine 模型拷贝到 exe 所在目录，防止运行时报“找不到 dll”
add_custom_command(TARGET trt_denoise_show POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TRT_DIR}/bin/nvinfer_10.dll" $<TARGET_FILE_DIR:trt_denoise_show>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TRT_DIR}/bin/nvinfer_dispatch_10.dll" $<TARGET_FILE_DIR:trt_denoise_show>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TRT_DIR}/bin/nvinfer_plugin_10.dll" $<TARGET_FILE_DIR:trt_denoise_show>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OCV_DIR}/x64/vc16/bin/opencv_world481.dll" $<TARGET_FILE_DIR:trt_denoise_show>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDA_DIR}/bin/cudart64_12.dll" $<TARGET_FILE_DIR:trt_denoise_show>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ENGINE_SOURCE_PATH}" $<TARGET_FILE_DIR:trt_denoise_show>
)