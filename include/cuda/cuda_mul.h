/*
 * @Author: liushijie
 * @Date: 2020-06-28 15:27:50
 * @LastEditTime: 2020-07-09 20:12:00
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/cuda/cuda_mul.h
 */
#pragma once

#include "dag/tensor.h"

namespace dl {
namespace cuda {
void CudaMul(const Tensor &x, const Tensor &y, Tensor *res);
void CudaMat(const Tensor &lhs, const Tensor &rhs, Tensor *res);
}
}