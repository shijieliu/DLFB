/*
 * @Author: liushijie
 * @Date: 2020-07-06 18:02:03
 * @LastEditTime: 2020-07-08 11:50:41
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/src/cuda_impl/add.cu
 */
#include "cuda/cuda_add.h"
#include "macro.h"
#include <cuda_runtime.h>

namespace dl {
namespace cuda {
__global__ void CudaAddKernal(float *x, float *y, float *res, int len) {
    int begin_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int read_offset = blockDim.x * gridDim.x;
    for (int i = begin_idx; i < len; i += read_offset) {
        res[i] = x[i] + y[i];
    }
}

void CudaAdd(const Tensor &x, const Tensor &y, Tensor *res) {
    CHECK_EQ(x.size(), res->size());
    CHECK_EQ(y.size(), res->size());

    int    len = x.size();
    float *devx, *devy, *dev_res;
    gpuErrchk(cudaMalloc((void **) &devx, sizeof(float) * len));
    gpuErrchk(cudaMalloc((void **) &devy, sizeof(float) * len));
    gpuErrchk(cudaMalloc((void **) &dev_res, sizeof(float) * len));

    gpuErrchk(cudaMemcpy(devx, x.data(), sizeof(float) * len,
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(devy, y.data(), sizeof(float) * len,
                         cudaMemcpyHostToDevice));
    CudaAddKernal<<<32, 1024>>>(devx, devy, dev_res, len);
    gpuErrchk(cudaMemcpy(res->data(), dev_res, sizeof(float) * len,
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(devx));
    gpuErrchk(cudaFree(devy));
    gpuErrchk(cudaFree(dev_res));
}
}
}