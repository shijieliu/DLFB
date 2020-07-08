/*
 * @Author: liushijie
 * @Date: 2020-07-08 11:09:48
 * @LastEditTime: 2020-07-08 11:12:49
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/cuda/cuda_conv.h
 */
#pragma once

#include "dag/tensor.h"

namespace dl {
namespace cuda {

void CudaConv2D(const Tensor &x, const Tensor &weight, Tensor *out,
                Tensor *flatten_x, int stride, int padding,
                const std::string &padding_mode);
}
}