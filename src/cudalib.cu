/*
 * @Author: liushijie
 * @Date: 2020-07-06 18:02:03
 * @LastEditTime: 2020-07-06 18:48:50
 * @LastEditors: liushijie
 * @Description: 
 * @FilePath: /LightLR/src/cudalib.cu
 */ 
#include <cuda_runtime.h>
#include "cuda/cudalib.h"
#include "macro.h"

namespace dl {
namespace cuda {
__global__ void CudaAddKernal(float *x, float *y, float *res, int len) {
    int begin_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int read_offset = blockDim.x * gridDim.x;
    for (int i = begin_idx; i < len; i += read_offset) {
        res[i] = x[i] + y[i];
    }
}

void CudaAdd(const float *x, const float *y, float *res, int len) {
    float *devx, *devy, *dev_res;
    gpuErrchk(cudaMalloc((void **)&devx, sizeof(float) * len));
    gpuErrchk(cudaMalloc((void **)&devy, sizeof(float) * len));
    gpuErrchk(cudaMalloc((void **)&dev_res, sizeof(float) * len));

    gpuErrchk(cudaMemcpy(devx, x, sizeof(float) * len, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(devy, y, sizeof(float) * len, cudaMemcpyHostToDevice));
    CudaAddKernal<<<BLOCK_SIZE, GRID_SIZE>>>(devx, devy, dev_res, len);
    gpuErrchk(cudaMemcpy(res, dev_res, sizeof(float) * len, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(devx));
    gpuErrchk(cudaFree(devy));
    gpuErrchk(cudaFree(dev_res));
}

// __global__ void conv_cuda(double *out, const double *filter, int K, int C, int FW, int FH,
//                           const double *paddedImage, int W, int H) {

//     int k = threadIdx.x + (blockDim.x * blockIdx.x);
//     int x = threadIdx.y + (blockDim.y * blockIdx.y);
//     int y = threadIdx.z + (blockDim.z * blockIdx.z);

//     auto outIndex = Expand(y, H, x, W, k);

//     double sum = 0.0;

// #pragma unroll 4
//     for (int c = 0; c < C; ++c) {
//         for (int j = 0; j < FH; ++j) {
//             for (int i = 0; i < FW; ++i) {
//                 auto filterIndex = Expand(FH - 1 - j, FH, FW - 1 - i, FW, c, C, k);
//                 auto imageIndex  = Expand(y + j, W + 2, x + i, H + 2, c);
//                 sum += (filter[filterIndex] * paddedImage[imageIndex]);
//             }
//         }
//     }
//     out[outIndex] = sum;
// }

// __global__ void conv_cuda_tiled(double *out, const double *filter, int K, int C, int FW, int FH,
//                                 const double *paddedImage, int W, int H) {

//     int k = threadIdx.x + (blockDim.x * blockIdx.x);
//     int x = threadIdx.y + (blockDim.y * blockIdx.y);
//     int y = threadIdx.z + (blockDim.z * blockIdx.z);

//     int tidx = threadIdx.y;
//     int tidy = threadIdx.z;

//     int X_BOUND = X_BLOCK + FW - 1;
//     int Y_BOUND = Y_BLOCK + FH - 1;

//     extern __shared__ double tile[];

//     for (int c = 0; c < C; ++c) {
//         tile[Expand(tidy, Y_BOUND, tidx, X_BOUND, c)] = paddedImage[Expand(y, W + 2, x, H + 2, c)];
//         // corner loads
//         if (tidx == X_BLOCK - 1 && tidy == Y_BLOCK - 1) {
//             for (int xx = 0; xx < FW; ++xx) {
//                 for (int yy = 0; yy < FH; ++yy) {
//                     if (xx == 0 and yy == 0)
//                         continue;
//                     tile[Expand(tidy + yy, Y_BOUND, tidx + xx, X_BOUND, c)] =
//                         paddedImage[Expand(y + yy, W + 2, x + xx, H + 2, c)];
//                 }
//             }
//         } // edge loads
//         else if (tidx == X_BLOCK - 1) {
//             for (int xx = 1; xx < FW; ++xx) {
//                 tile[Expand(tidy, Y_BOUND, tidx + xx, X_BOUND, c)] =
//                     paddedImage[Expand(y, W + 2, x + xx, H + 2, c)];
//             }
//         } else if (tidy == Y_BLOCK - 1) {
//             for (int yy = 1; yy < FH; ++yy) {
//                 tile[Expand(tidy + yy, Y_BOUND, tidx, X_BOUND, c)] =
//                     paddedImage[Expand(y + yy, W + 2, x, H + 2, c)];
//             }
//         }
//     }
//     __syncthreads();

//     auto   outIndex = Expand(y, H, x, W, k);
//     double sum      = 0.0;
// #pragma unroll 4
//     for (int c = 0; c < C; ++c) {
//         for (int j = 0; j < FH; ++j) {
//             for (int i = 0; i < FW; ++i) {
//                 auto filterIndex = Expand(FH - 1 - j, FH, FW - 1 - i, FW, c, C, k);
//                 auto imageIndex  = Expand(tidy + j, Y_BOUND, tidx + i, X_BOUND, c);
//                 sum += (filter[filterIndex] * tile[imageIndex]);
//             }
//         }
//     }
//     out[outIndex] = sum;
// }
}
}