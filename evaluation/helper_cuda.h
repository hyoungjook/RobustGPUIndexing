// for compatibility with DyCuckoo
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>

inline void __check_cuda_errors(cudaError_t status, const char* file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(status));
    std::abort();
  }
}

#define checkCudaErrors(call) __check_cuda_errors((call), __FILE__, __LINE__)
