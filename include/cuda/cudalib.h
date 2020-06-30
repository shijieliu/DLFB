/*
 * @Author: liushijie
 * @Date: 2020-06-28 15:27:50
 * @LastEditTime: 2020-06-28 15:27:51
 * @LastEditors: liushijie
 * @Description: 
 * @FilePath: /LightLR/include/cuda/cudalib.h
 */ 
#ifndef CUDA_H
#define CUDA_H

namespace dl {
namespace cuda {
const int BLOCK_SIZE = 32;
const int GRID_SIZE  = 32;
void CudaAdd(const float *x, const float *y, float *res, int size);

}
}

#endif