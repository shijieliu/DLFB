/*
 * @Author: liushijie
 * @Date: 2020-06-28 15:27:50
 * @LastEditTime: 2020-07-08 11:50:56
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/cuda/cuda_add.h
 */
#pragma once

#include "dag/tensor.h"

namespace dl {
namespace cuda {
void CudaAdd(const Tensor &x, const Tensor &y, Tensor *res);
}
}