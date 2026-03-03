cmake_minimum_required(VERSION 3.15)
project(trt_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# --- 1. 更新为你新电脑的绝对路径 ---
set(TRT_DIR "D:/TensorRT-10.15.1.29")
set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1")
set(OCV_DIR "D:/opencv/build")

# 更新 Engine 的名字为 3060 版本
set(ENGINE_SOURCE_PATH "${PROJECT_SOURCE_DIR}/../checkpoint/model_trt_3060_fp16.engine")

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_executable(trt_denoise_show ${SRC_FILES})

target_include_directories(trt_denoise_show PRIVATE
    "${TRT_DIR}/include"
    "${PROJECT_SOURCE_DIR}/include"
    "${CUDA_DIR}/include"
    "${OCV_DIR}/include"
)

# --- 2. 链接 TensorRT 10 库 ---
target_link_libraries(trt_denoise_show PRIVATE
    "${TRT_DIR}/lib/nvinfer_10.lib"
    "${TRT_DIR}/lib/nvinfer_plugin_10.lib"
    "${TRT_DIR}/lib/nvinfer_dispatch_10.lib"
    "${CUDA_DIR}/lib/x64/cudart.lib"
    "${CUDA_DIR}/lib/x64/cuda.lib"
    "${OCV_DIR}/x64/vc16/lib/opencv_world481.lib"
)

if(MSVC)
    target_compile_options(trt_denoise_show PRIVATE /MD /W3 /utf-8 /wd4996)
endif()

# --- 3. 拷贝 DLL (重点修正了 CUDA 12 的 dll 名字) ---
add_custom_command(TARGET trt_denoise_show POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TRT_DIR}/bin/nvinfer_10.dll" $<TARGET_FILE_DIR:trt_denoise_show>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TRT_DIR}/bin/nvinfer_dispatch_10.dll" $<TARGET_FILE_DIR:trt_denoise_show>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TRT_DIR}/bin/nvinfer_plugin_10.dll" $<TARGET_FILE_DIR:trt_denoise_show>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OCV_DIR}/x64/vc16/bin/opencv_world481.dll" $<TARGET_FILE_DIR:trt_denoise_show>
    
    # 【修复重点】：cudart64_110.dll 改为 cudart64_12.dll
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDA_DIR}/bin/cudart64_12.dll" $<TARGET_FILE_DIR:trt_denoise_show>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ENGINE_SOURCE_PATH}" $<TARGET_FILE_DIR:trt_denoise_show>
)