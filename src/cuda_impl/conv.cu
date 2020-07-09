/*
 * @Author: liushijie
 * @Date: 2020-07-08 11:10:20
 * @LastEditTime: 2020-07-09 19:04:11
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/src/cuda_impl/conv.cu
 */

#include "cuda/cuda_conv.h"
#include "macro.h"
#include <cuda_runtime.h>

namespace dl {
namespace cuda {
__global__ void CudaConv2DKernal(float *x, float *weight, float *res, int h,
                                 int w, int kernel_size, int c_in, int stride) {
    // dim3 grid_size(c_out, n);
    // dim3 block_size(w_out, h_out, c_in);

    int c_out = gridDim.x;
    int w_out = blockDim.x;
    int h_out = blockDim.y;

    int out_idx = Expand(threadIdx.x, w_out, threadIdx.y, h_out, blockIdx.x,
                         c_out, blockIdx.y);
    float sum = 0.0f;
    for (int k1 = 0; k1 < kernel_size; ++k1) {
        for (int k2 = 0; k2 < kernel_size; ++k2) {
            for (int c = 0; c < c_in; ++c) {
                sum += weight[Expand(k1, kernel_size, k2, kernel_size,
                                              c, c_in, blockIdx.x)] *
                                x[Expand(threadIdx.x * stride + k1, w,
                                         threadIdx.y * stride + k2, h, c, c_in,
                                         blockIdx.y)];
            }
        }
    }
    res[out_idx] = sum;
}

void CudaConv2D(const Tensor &x, const Tensor &weight, Tensor *out,
                Tensor *flatten_x, int stride, int padding,
                const std::string &padding_mode) {
    CHECK_EQ(x.shape().size(), 4);
    CHECK_EQ(weight.shape().size(), 4);
    LOG_DEBUG("\n\tTensor CudaConv2D args:\n\t\tx shape:(%d, %d, %d, "
              "%d)\n\t\tweight shape:(%d, %d, %d, "
              "%d)\n\t\tstride:%d\n\t\tpadding:%d",
              x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3],
              weight.shape()[0], weight.shape()[1], weight.shape()[2],
              weight.shape()[3], stride, padding);

    CHECK_EQ(x.shape()[1], weight.shape()[1]);

    int c_out = weight.shape()[0];
    int c_in  = weight.shape()[1];
    int k     = weight.shape()[2];
    int n     = x.shape()[0];
    int h     = x.shape()[2];
    int w     = x.shape()[3];
    int h_out = (h + 2 * padding - k) / stride + 1;
    int w_out = (w + 2 * padding - k) / stride + 1;
    LOG_DEBUG("(h_out: %d, w_out: %d)", h_out, w_out);
    CHECK_EQ(out->shape()[0], n);
    CHECK_EQ(out->shape()[1], c_out);
    CHECK_EQ(out->shape()[2], h_out);
    CHECK_EQ(out->shape()[3], w_out);

    float *dev_x, *dev_weight, *dev_res;
    gpuErrchk(cudaMalloc(&dev_x, sizeof(float) * x.size()));
    gpuErrchk(cudaMalloc(&dev_weight, sizeof(float) * weight.size()));
    gpuErrchk(cudaMalloc(&dev_res, sizeof(float) * n * c_out * h_out * w_out));
    gpuErrchk(cudaMemcpy(dev_x, x.data(), sizeof(float) * x.size(),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_weight, weight.data(),
                         sizeof(float) * weight.size(),
                         cudaMemcpyHostToDevice));

    dim3 grid_size(c_out, n);
    dim3 block_size(w_out, h_out);
    CudaConv2DKernal<<<grid_size, block_size>>>(dev_x, dev_weight, dev_res, h,
                                                w, k, c_in, stride);
    gpuErrchk(cudaMemcpy(out->data(), dev_res, sizeof(float) * out->size(),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(dev_x));
    gpuErrchk(cudaFree(dev_weight));
    gpuErrchk(cudaFree(dev_res));
}
}
}