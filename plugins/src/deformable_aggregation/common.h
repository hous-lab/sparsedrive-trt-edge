#pragma once
#include <iostream>
#include <algorithm>

#if DEBUG
#define print_log(...) {\
    char str[100];\
    sprintf(str, __VA_ARGS__);\
    std::cout << "PLUGIN TRACE --> " << "[" \
              << __FILE__ << ":" << __LINE__ \
              << ", " << __FUNCTION__ << "] " << str << std::endl;\
    }
#else
#define print_log(...)
#endif

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = DIVUP(N, THREADS_PER_BLOCK);
  int max_block_num = 4096;
  return std::min(optimal_block_num, max_block_num);
}

#define cudaCheckError()                                                               \
  {                                                                                    \
    cudaError_t e = cudaGetLastError();                                                \
    if (e != cudaSuccess) {                                                            \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(0);                                                                         \
    }                                                                                  \
  }

#if NV_TENSORRT_MAJOR > 7
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOEXCEPT
#endif
