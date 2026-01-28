cmake_minimum_required(VERSION 3.15)
project(trt_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

set(TRT_DIR "C:/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6")
set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
set(OCV_DIR "C:/opencv/build")
set(ENGINE_SOURCE_PATH "${PROJECT_SOURCE_DIR}/../checkpoint/model_2_trt_2060_fp16.engine")

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
add_executable(trt_denoise_show ${SRC_FILES})

target_include_directories(trt_denoise_show PRIVATE
    "${TRT_DIR}/include"
    "${PROJECT_SOURCE_DIR}/include"
    "${CUDA_DIR}/include"
    "${OCV_DIR}/include"
)

target_link_libraries(trt_denoise_show PRIVATE
    "${TRT_DIR}/lib/nvinfer.lib"
    "${TRT_DIR}/lib/nvinfer_plugin.lib"
    "${CUDA_DIR}/lib/x64/cudart.lib"
    "${CUDA_DIR}/lib/x64/cuda.lib"
    "${OCV_DIR}/x64/vc16/lib/opencv_world481.lib"
)

if(MSVC)
    target_compile_options(trt_denoise_show PRIVATE /MD /W3 /utf-8 /wd4996)
endif()

add_custom_command(TARGET trt_denoise_show POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TRT_DIR}/lib/nvinfer.dll" $<TARGET_FILE_DIR:trt_denoise_show>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TRT_DIR}/lib/nvinfer_plugin.dll" $<TARGET_FILE_DIR:trt_denoise_show>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OCV_DIR}/x64/vc16/bin/opencv_world481.dll" $<TARGET_FILE_DIR:trt_denoise_show>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDA_DIR}/bin/cudart64_110.dll" $<TARGET_FILE_DIR:trt_denoise_show>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ENGINE_SOURCE_PATH}" $<TARGET_FILE_DIR:trt_denoise_show>
)