/*
 * @Author: liushijie
 * @Date: 2020-07-06 18:02:03
 * @LastEditTime: 2020-07-09 20:39:46
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/src/cuda_impl/mul.cu
 */
#include "cuda/cuda_mul.h"
#include "macro.h"
#include <cuda_runtime.h>

namespace dl {
namespace cuda {
__global__ void CudaMulKernal(float *x, float *y, float *res, int len) {
    int begin_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int read_offset = blockDim.x * gridDim.x;
    for (int i = begin_idx; i < len; i += read_offset) {
        res[i] = x[i] * y[i];
    }
}

void CudaMul(const Tensor &x, const Tensor &y, Tensor *res) {
    CHECK_EQ(x.size(), res->size());
    CHECK_EQ(y.size(), res->size());

    int    len = x.size();
    float *devx, *devy, *dev_res;
    gpuErrchk(cudaMalloc(&devx, sizeof(float) * len));
    gpuErrchk(cudaMalloc(&devy, sizeof(float) * len));
    gpuErrchk(cudaMalloc(&dev_res, sizeof(float) * len));

    gpuErrchk(cudaMemcpy(devx, x.data(), sizeof(float) * len,
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(devy, y.data(), sizeof(float) * len,
                         cudaMemcpyHostToDevice));
    CudaMulKernal<<<32, 1024>>>(devx, devy, dev_res, len);
    gpuErrchk(cudaMemcpy(res->data(), dev_res, sizeof(float) * len,
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(devx));
    gpuErrchk(cudaFree(devy));
    gpuErrchk(cudaFree(dev_res));
}

template <size_t TileSize>
__global__ void CudaMatKernal(float *x, float *y, float *res, int row, int col,
                              int len) {
    __shared__ float shared_x[TileSize][TileSize];
    __shared__ float shared_y[TileSize][TileSize];

    int   tx    = threadIdx.x;
    int   ty    = threadIdx.y;
    int   bx    = blockIdx.x;
    int   by    = blockIdx.y;
    int   r     = Expand(ty, TileSize, by);
    int   c     = Expand(tx, TileSize, bx);
    float value = 0.;
    for (int m = 0; m < (len - 1) / TileSize + 1; ++m) {
        if (r < row && Expand(tx, TileSize, m) < len) {
            shared_x[tx][ty] =
                x[Expand(Expand(tx, TileSize, m), len, r)]; // x shape(row, len)
                                                            // x idx (r,
                                                            // Expand(tx,
                                                            // TileSize, m))
        } else {
            shared_x[tx][ty] = 0;
        }
        if (c < col && Expand(ty, TileSize, m) < len) {
            shared_y[tx][ty] =
                y[Expand(c, col, Expand(tx, TileSize, m))]; // y shape (len,
                                                            // col) y idx
                                                            // (Expand(tx,
                                                            // TileSize, m) , c)
        } else {
            shared_y[tx][ty] = 0;
        }

        __syncthreads();
        for (int k = 0; k < TileSize; ++k) {
            value += shared_x[tx][k] * shared_y[k][ty];
        }
        __syncthreads();
    }
    res[Expand(c, col, r)] = value;
}

void CudaMat(const Tensor &lhs, const Tensor &rhs, Tensor *res) {
    CHECK_EQ(lhs.shape().size(), 2);
    CHECK_EQ(lhs.shape()[1], rhs.shape()[0]);

    int row = lhs.shape()[0];
    int col = rhs.shape()[1];
    int len = lhs.shape()[1];
    CHECK_EQ(res->size(), row * col);

    float *devx, *devy, *dev_res;
    gpuErrchk(cudaMalloc(&devx, sizeof(float) * lhs.size()));
    gpuErrchk(cudaMalloc(&devy, sizeof(float) * rhs.size()));
    gpuErrchk(cudaMalloc(&dev_res, sizeof(float) * res->size()));

    gpuErrchk(cudaMemcpy(devx, lhs.data(), sizeof(float) * lhs.size(),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(devy, rhs.data(), sizeof(float) * rhs.size(),
                         cudaMemcpyHostToDevice));

    constexpr int tile_size = 16;
    dim3 grid_size((row - 1) / tile_size + 1, (col - 1) / tile_size + 1);
    dim3 block_size(tile_size, tile_size);
    CudaMatKernal<tile_size><<<grid_size, block_size>>>(devx, devy, dev_res, row, col, len);
    gpuErrchk(cudaMemcpy(res->data(), dev_res, sizeof(float) * res->size(),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(devx));
    gpuErrchk(cudaFree(devy));
    gpuErrchk(cudaFree(dev_res));
}
}
}