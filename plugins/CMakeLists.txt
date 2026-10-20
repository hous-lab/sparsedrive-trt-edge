cmake_minimum_required(VERSION 3.18)
project(sparsedrive_deploy LANGUAGES C CXX CUDA)

set(TENSORRT_ROOT "$ENV{TRT_ROOT}" CACHE PATH "TensorRT root directory")
set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA root directory")
set(ARCH "89" CACHE STRING "GPU architecture (89 for 4090, 87 for Orin)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xptxas=-v")

message(STATUS "TRT_ROOT: ${TENSORRT_ROOT}")
message(STATUS "CUDA_ROOT: ${CUDA_ROOT}")
message(STATUS "ARCH: sm_${ARCH}")

include_directories(
    ${CUDA_ROOT}/include
    ${TENSORRT_ROOT}/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deformable_aggregation)

link_directories(${TENSORRT_ROOT}/lib/ ${CUDA_ROOT}/lib64)

set(plugins_cu_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deformable_aggregation/deformable_aggregation_kernel.cu)

add_library(plugins_cu STATIC ${plugins_cu_srcs})
set_target_properties(plugins_cu PROPERTIES
    CUDA_ARCHITECTURES ${ARCH}
    POSITION_INDEPENDENT_CODE ON)

add_library(plugins SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/logger_def.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deformable_aggregation/deformable_aggregation.cpp)

target_link_libraries(plugins plugins_cu nvinfer cublas cudart)
